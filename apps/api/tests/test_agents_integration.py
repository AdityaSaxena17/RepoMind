"""Opt-in specialist mechanics against the isolated evaluation database."""

import asyncio
import os
import uuid

import asyncpg
import pytest
from test_agents import ScriptedModel, call, summary
from test_evals import FakeEmbeddings

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
