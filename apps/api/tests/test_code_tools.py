import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app import code_tools
from app.code_tools import CodeTools
from app.tool_registry import RequestScope, ToolRegistry

SHA = "a" * 40


class Embedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class Database:
    def __init__(self) -> None:
        self.allowed = True
        self.sha = SHA
        self.files = {"app.py": "one\ntwo\nthree\nfour\nfive\nsix"}
        self.connections: list[Connection] = []
        self.active = 0
        self.maximum = 0
        self.revision = datetime(2026, 10, 15, tzinfo=timezone.utc)
        self.users = {"user"}
        self.queries: list[tuple[Any, ...]] = []


class Connection:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.in_transaction = False
        self.snapshot = SHA

    @asynccontextmanager
    async def transaction(self, **kwargs: Any) -> AsyncIterator[None]:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        assert not self.in_transaction, "Shared connection used concurrently."
        self.in_transaction = True
        self.snapshot = self.database.sha
        self.database.active += 1
        self.database.maximum = max(self.database.maximum, self.database.active)
        try:
            await asyncio.sleep(0)
            yield
        finally:
            self.database.active -= 1
            self.in_transaction = False

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        assert self.in_transaction
        self.database.queries.append((sql, *args))
        await asyncio.sleep(0)
        if 'FROM "Repository"' in sql:
            assert 'ur."accessStatus"' in sql and 'r."accessStatus"' in sql
            return (
                {
                    "indexedCommitSha": self.snapshot,
                    "indexStatus": "READY",
                    "htmlUrl": "https://github.com/example/project",
                    "lastIndexedAt": self.database.revision,
                }
                if self.database.allowed
                and args[0] == "repo"
                and args[1] in self.database.users
                else None
            )
        assert args[0] == "repo"
        content = self.database.files.get(args[1])
        return {"content": content} if content is not None else None


class Pool:
    def __init__(self, database: Database) -> None:
        self.database = database

    @asynccontextmanager
    async def acquire(self, **kwargs: Any) -> AsyncIterator[Connection]:
        assert kwargs == {"timeout": 10}
        connection = Connection(self.database)
        self.database.connections.append(connection)
        yield connection
        assert not connection.in_transaction


def setup_tools(
    database: Database, pooled: bool = False
) -> tuple[ToolRegistry, CodeTools]:
    scope = RequestScope(
        user_id="user", repository_id="repo", capabilities=frozenset({"code.read"})
    )
    registry = ToolRegistry(scope)
    arguments = (
        {"pool": Pool(database)} if pooled else {"connection": Connection(database)}
    )
    return registry, CodeTools(registry, Embedding(), SHA, **arguments)


def scripted_retriever(
    monkeypatch: pytest.MonkeyPatch, database: Database
) -> list[str]:
    queries: list[str] = []

    class Retriever:
        def __init__(self, **kwargs: Any) -> None:
            self.connection = kwargs["connection"]
            assert kwargs["repository_id"] == "repo"

        async def ainvoke(self, query: str, config: Any = None) -> list[Document]:
            assert self.connection.in_transaction
            queries.append(query)
            await asyncio.sleep(0)
            if query == "empty":
                return []
            first, last = (1, 2) if query != "followup" else (3, 4)
            return [
                Document(
                    page_content="\n".join(
                        database.files["app.py"].splitlines()[first - 1 : last]
                    ),
                    metadata={"path": "app.py", "start_line": first, "end_line": last},
                )
            ]

    monkeypatch.setattr(code_tools, "RepoMindParentHybridRetriever", Retriever)
    return queries


def test_search_file_reads_share_stable_catalog_and_first_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    queries = scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, catalog = setup_tools(database)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            search, read = tools
            first = await search.ainvoke({"query": "initial"})
            same = await read.ainvoke(
                {"path": "app.py", "start_line": 1, "end_line": 2}
            )
            assert json.loads(first)["sources"] == json.loads(same)["sources"]
            await search.ainvoke({"query": "followup"})
            again = await search.ainvoke({"query": "initial"})
            assert again == first
        assert queries == ["initial", "followup", "initial"]
        assert len(catalog.evidence) == 2
        assert [source.source_index for source in catalog.evidence.values()] == [1, 2]
        assert catalog.first_query == "initial"
        assert [source.start_line for source in catalog.first_pass] == [1]
        assert catalog.first_pass[0].url.endswith(f"/{SHA}/app.py#L1-L2")
        assert database.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("pooled", [False, True])
def test_connection_ownership_and_parallel_local_calls(
    monkeypatch: pytest.MonkeyPatch, pooled: bool
) -> None:
    database = Database()
    scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, catalog = setup_tools(database, pooled)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await asyncio.gather(
                *[
                    tools[0].ainvoke({"query": query})
                    for query in ("initial", "followup")
                ]
            )
        assert database.maximum == (2 if pooled else 1)
        assert database.active == 0
        assert len(catalog.evidence) == 2
        if pooled:
            assert len(database.connections) == 4

    asyncio.run(run())


@pytest.mark.parametrize("revoked", [False, True])
def test_revocation_and_snapshot_change_prevent_new_evidence(
    monkeypatch: pytest.MonkeyPatch, revoked: bool
) -> None:
    database = Database()
    queries = scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, catalog = setup_tools(database)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0].ainvoke({"query": "initial"})
            if revoked:
                database.allowed = False
            else:
                database.sha = "b" * 40
            result = await tools[0].ainvoke(
                {
                    "name": tools[0].name,
                    "args": {"query": "followup"},
                    "id": "denied",
                    "type": "tool_call",
                }
            )
            assert result.status == "error"
        assert queries == ["initial"]
        assert len(catalog.evidence) == 1
        assert database.active == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "app.py", "start_line": 0},
        {"path": "app.py", "start_line": True},
        {"path": "app.py", "start_line": 4, "end_line": 1},
        {"path": "app.py", "end_line": 251},
        {"path": "app.py", "repository_id": "other"},
        {"path": "missing.py"},
        {"path": "app.py", "start_line": 100},
    ],
)
def test_file_arguments_fail_without_fabricating_evidence(
    arguments: dict[str, Any],
) -> None:
    async def run() -> None:
        registry, catalog = setup_tools(Database())
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            result = await tools[1].ainvoke(
                {
                    "name": tools[1].name,
                    "args": arguments,
                    "id": "bad",
                    "type": "tool_call",
                }
            )
            assert result.status == "error"
            assert result.tool_call_id == "bad"
            if result.additional_kwargs.get("repomind_error_kind") == "validation":
                assert "250" in result.text and "Split" in result.text
            assert len(result.text) < 500 and "Traceback" not in result.text
        assert catalog.evidence == {}

    asyncio.run(run())


def test_read_through_known_eof_is_complete() -> None:
    async def run() -> None:
        registry, catalog = setup_tools(Database())
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            result = json.loads(
                await tools[1].ainvoke({"path": "app.py", "end_line": 10})
            )
        assert result["file_line_count"] == 6
        assert result["sources"][0]["end_line"] == 6
        assert result["missing_information"] == []

    asyncio.run(run())


def test_empty_reads_and_whole_line_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    database = Database()
    database.files.update({"empty.py": "", "large.py": "a" * 8000 + "\n" + "b" * 8000})
    scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, catalog = setup_tools(database)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0].ainvoke({"query": "empty"})
            empty = await tools[1].ainvoke({"path": "empty.py"})
            assert '"sources": []' in empty
            truncated = await tools[1].ainvoke(
                {
                    "name": tools[1].name,
                    "args": {"path": "large.py", "end_line": 2},
                    "id": "large",
                    "type": "tool_call",
                }
            )
            assert truncated.artifact["sources"][0]["end_line"] == 1
            assert len(truncated.artifact["sources"][0]["content"]) == 8000
            assert truncated.artifact["missing_information"]
        assert catalog.first_pass == []
        assert catalog.first_query == "empty"
        assert database.active == 0

    asyncio.run(run())
