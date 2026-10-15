"""Opt-in specialist mechanics against the isolated evaluation database."""

import asyncio
import json
import os
import uuid
from typing import Any

import asyncpg
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_agents import ScriptedModel, call, summary
from test_evals import FakeEmbeddings
from test_repository_graph import RouterModel

from app import worker
from app.code_tools import CodeTools
from app.config import get_settings, normalize_asyncpg_database_url
from app.evals.execution import (
    Benchmark,
    SpecializedAgentRunner,
    connect_evaluation_database,
    fixture_tools,
    prepare_index,
)
from app.evals.models import Dataset, FixtureCall, Fixtures, Question
from app.tool_registry import RequestScope, ToolRegistry

pytestmark = pytest.mark.skipif(
    os.getenv("REPOMIND_EVAL_INTEGRATION") != "1",
    reason="Opt in with evaluation-only database credentials.",
)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_pooled_reads_reindexing_revocation_and_eval_access_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_EMBEDDINGS_MODEL", "eval-test-deterministic-1536")
    get_settings.cache_clear()

    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        class Embeddings(FakeEmbeddings):
            gated = False

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [
                    [1.0] + [0.0] * (get_settings().embedding_dimensions - 1)
                    for _ in texts
                ]

            async def aembed_query(self, text: str) -> list[float]:
                if self.gated:
                    started.set()
                    await release.wait()
                return self.embed_documents([text])[0]

        embeddings = Embeddings()
        monkeypatch.setattr(worker, "create_embedding_model", lambda: embeddings)
        sha, new_sha = "0" * 40, "b" * 40
        content = "def unique_agent_symbol():\n    return 42\n"
        dataset = Dataset(
            version=f"agents-integration-{uuid.uuid4().hex}",
            repository="fixture/agents",
            corpus_commit_sha=sha,
            annotation_policy="Integration mechanics only.",
            questions=[
                Question(
                    id="code",
                    category="single_file",
                    evidence_scope="CODE",
                    question="unique_agent_symbol",
                    review_note="Mechanics",
                )
            ],
        )
        benchmark = Benchmark(
            dataset,
            Fixtures(version="test", provenance="test", scenarios={"none": []}),
            {"app.py": content},
            {"app.py": sha},
        )
        connection = await connect_evaluation_database()
        pool = None
        pending = None
        repository_id = ""
        user_id = f"eval_test_{uuid.uuid4().hex}"
        try:
            repository_id = await prepare_index(connection, benchmark, embeddings)
            await connection.execute(
                'INSERT INTO "User" ("id", "clerkId", "updatedAt") '
                "VALUES ($1,$1,now())",
                user_id,
            )
            await connection.execute(
                'INSERT INTO "UserRepository" '
                '("id", "userId", "repositoryId", "updatedAt") '
                "VALUES ($1,$1,$2,now())",
                user_id,
                repository_id,
            )
            before = await connection.fetchval('SELECT count(*) FROM "User"')
            code_runner = SpecializedAgentRunner(
                connection,
                embeddings,
                repository_id,
                sha,
                dataset.repository,
                "CODE",
                ScriptedModel(
                    responses=[
                        call(
                            "search_repository_code", {"query": "unique_agent_symbol"}
                        ),
                        summary(),
                    ]
                ),
                [],
                [],
            )
            code_result = await code_runner.run("Explain unique_agent_symbol.")
            assert len(code_result.final_contexts) == 1
            assert code_result.evidence_seen[0].content == content.rstrip("\n")
            assert code_runner.last_findings.status == "complete"
            url = "https://github.com/fixture/agents/issues/1"
            fixtures = Fixtures(
                version="test",
                provenance="synthetic",
                scenarios={
                    "one": [
                        FixtureCall(
                            id="one",
                            tool="issue_read",
                            arguments=[{"issue_number": 1}],
                            result={
                                "issue": {
                                    "url": url,
                                    "title": "An issue",
                                    "state": "open",
                                }
                            },
                        )
                    ]
                },
            )
            records = []
            github_runner = SpecializedAgentRunner(
                connection,
                embeddings,
                repository_id,
                sha,
                dataset.repository,
                "GITHUB",
                ScriptedModel(
                    responses=[
                        call("issue_read", {"issue_number": 1}),
                        summary(url, [url]),
                    ]
                ),
                await fixture_tools(fixtures, "one", records),
                records,
            )
            github_result = await github_runner.run("Read issue 1.")
            assert github_result.final_contexts == []
            assert github_result.tool_executions[0].arguments["method"] == "get"
            assert github_result.tool_executions[0].arguments["per_page"] == 10
            assert github_runner.last_findings.status == "complete"
            failed_runner = SpecializedAgentRunner(
                connection,
                embeddings,
                repository_id,
                sha,
                dataset.repository,
                "CODE",
                ScriptedModel(responses=[summary(sources=["invented"])]),
                [],
                [],
            )
            with pytest.raises(RuntimeError, match="Specialized investigation failed"):
                await failed_runner.run("question")
            assert await connection.fetchval('SELECT count(*) FROM "User"') == before

            class RoleModel(ScriptedModel):
                broken_summary: bool = False
                missing_fixture: bool = False

                def _generate(
                    self, messages: Any, *args: Any, **kwargs: Any
                ) -> ChatResult:
                    prompt = messages[0].text
                    initial = not any(isinstance(m, AIMessage) for m in messages)
                    if prompt.startswith("You are RepoMind's Code Agent."):
                        message = (
                            call(
                                "search_repository_code",
                                {"query": "unique_agent_symbol"},
                                "code",
                            )
                            if initial
                            else summary()
                        )
                    elif prompt.startswith("You are RepoMind's GitHub Agent."):
                        message = (
                            call(
                                "issue_read",
                                {
                                    "issue_number": "999"
                                    if self.missing_fixture
                                    else "1"
                                },
                                "github",
                            )
                            if initial
                            else summary(url, [url])
                        )
                    else:
                        message = summary(
                            f"Code [1] and issue {url}.",
                            ["invented"] if self.broken_summary else ["1", url],
                        )
                    return ChatResult(generations=[ChatGeneration(message=message)])

            for scope, strategy, broken, missing_fixture in (
                ("CODE", None, False, False),
                ("GITHUB", None, False, False),
                ("MIXED", "PARALLEL", False, False),
                ("MIXED", "GITHUB_THEN_CODE", False, False),
                ("MIXED", "PARALLEL", True, False),
                ("MIXED", "PARALLEL", False, True),
            ):
                graph_records = []
                graph_runner = SpecializedAgentRunner(
                    connection,
                    embeddings,
                    repository_id,
                    sha,
                    dataset.repository,
                    "GRAPH",
                    RoleModel(
                        responses=[],
                        broken_summary=broken,
                        missing_fixture=missing_fixture,
                    ),
                    await fixture_tools(fixtures, "one", graph_records),
                    graph_records,
                    router_model=RouterModel(
                        responses=[
                            AIMessage(
                                content=json.dumps(
                                    {
                                        "scope": scope,
                                        "clarification": None,
                                        "mixed_strategy": strategy,
                                    }
                                )
                            )
                        ]
                    ),
                )
                if broken or missing_fixture:
                    with pytest.raises(
                        RuntimeError, match="Specialized investigation failed"
                    ):
                        await graph_runner.run("Explain the indexed code and issue.")
                    graph_result = graph_runner.last_result
                    assert graph_result.evidence_seen and not graph_result.answer
                else:
                    graph_result = await graph_runner.run(
                        "Explain the indexed code and issue."
                    )
                assert graph_runner.last_findings.status == (
                    "failed" if broken or missing_fixture else "complete"
                ), graph_runner.last_findings.errors
                assert len(graph_result.final_contexts) == (
                    0 if scope == "GITHUB" else 1
                )
                if scope != "CODE":
                    executed = next(
                        r
                        for r in graph_result.tool_executions
                        if r.name == "issue_read"
                    )
                    assert executed.arguments["method"] == "get"
                    assert executed.arguments["per_page"] == 10
                    assert executed.arguments["issue_number"] == (
                        999 if missing_fixture else 1
                    )
                    assert executed.fixture_miss == missing_fixture
                    if not missing_fixture:
                        assert graph_result.citation_catalog[url] == executed.id
                assert (
                    await connection.fetchval('SELECT count(*) FROM "User"') == before
                )

            cancel_started = asyncio.Event()

            class BlockingModel(RoleModel):
                async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
                    cancel_started.set()
                    await asyncio.sleep(10)

            cancel_runner = SpecializedAgentRunner(
                connection,
                embeddings,
                repository_id,
                sha,
                dataset.repository,
                "GRAPH",
                BlockingModel(responses=[]),
                [],
                [],
                router_model=RouterModel(
                    responses=[
                        AIMessage(
                            content=json.dumps(
                                {
                                    "scope": "CODE",
                                    "clarification": None,
                                }
                            )
                        )
                    ]
                ),
            )
            cancelled = asyncio.create_task(cancel_runner.run("Question"))
            await asyncio.wait_for(cancel_started.wait(), 5)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert await connection.fetchval('SELECT count(*) FROM "User"') == before

            url_config = os.environ["EVAL_DATABASE_URL"]
            pool = await asyncpg.create_pool(
                normalize_asyncpg_database_url(url_config),
                min_size=1,
                max_size=3,
                command_timeout=10,
            )
            registry = ToolRegistry(
                RequestScope(
                    user_id=user_id,
                    repository_id=repository_id,
                    capabilities=frozenset({"code.read"}),
                )
            )
            catalog = CodeTools(registry, embeddings, sha, pool=pool, rerank=False)
            async with registry.scoped_tools(frozenset({"code.read"})) as tools:
                await asyncio.gather(
                    tools[0].ainvoke({"query": "unique_agent_symbol"}),
                    tools[1].ainvoke({"path": "app.py"}),
                )
                assert len(catalog.evidence) == 1
                embeddings.gated = True
                pending = asyncio.create_task(
                    tools[0].ainvoke(
                        {
                            "name": tools[0].name,
                            "args": {"query": "unique_agent_symbol"},
                            "id": "snapshot",
                            "type": "tool_call",
                        }
                    )
                )
                await asyncio.wait_for(started.wait(), 5)
                new_content = content.replace("42", "84")
                chunks = worker.chunk_file("app.py", new_content)
                await worker.promote_index(
                    connection,
                    repository_id,
                    repository_id + "_job",
                    new_sha,
                    [
                        {
                            "path": "app.py",
                            "blob_sha": new_sha,
                            "content_sha": "new",
                            "byte_size": len(new_content),
                            "line_count": 2,
                            "content": new_content,
                        }
                    ],
                    chunks,
                    embeddings.embed_documents([d.page_content for d in chunks]),
                )
                release.set()
                result = await pending
                assert result.status == "success"
                assert "42" in result.artifact["sources"][0]["content"]
                stale = await tools[1].ainvoke(
                    {
                        "name": tools[1].name,
                        "args": {"path": "app.py"},
                        "id": "stale",
                        "type": "tool_call",
                    }
                )
                assert stale.status == "error"
                assert "snapshot changed" in stale.content
                await connection.execute(
                    'UPDATE "UserRepository" SET "accessStatus"=\'REVOKED\' '
                    'WHERE "userId"=$1',
                    user_id,
                )
                denied = await tools[1].ainvoke(
                    {
                        "name": tools[1].name,
                        "args": {"path": "app.py"},
                        "id": "revoked",
                        "type": "tool_call",
                    }
                )
                assert denied.status == "error"
                assert "not authorized" in denied.content
            assert all(e.corpus_commit_sha == sha for e in catalog.evidence.values())
        finally:
            release.set()
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if pool is not None:
                await pool.close()
            if repository_id:
                await connection.execute(
                    'DELETE FROM "Repository" WHERE "id"=$1', repository_id
                )
            await connection.execute('DELETE FROM "User" WHERE "id"=$1', user_id)
            await connection.close()

    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()
