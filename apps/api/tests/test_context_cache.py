import asyncio
import json
import logging
from datetime import timedelta
from typing import Any

import pytest
from redis.exceptions import RedisError
from test_code_tools import Database, Pool, scripted_retriever

from app import code_tools as module
from app.code_tools import CodeTools
from app.config import get_settings
from app.tool_registry import RequestScope, ToolRegistry


class Redis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gets = self.sets = 0
        self.failure: str | None = None
        self.canceled = False

    async def get(self, key: str) -> bytes | None:
        self.gets += 1
        if self.failure == "read":
            raise RedisError("Private query must not appear in diagnostics.")
        if self.failure == "timeout":
            try:
                await asyncio.sleep(10)
            finally:
                self.canceled = True
        return self.values.get(key)

    async def set(self, key: str, value: bytes, *, ex: int) -> None:
        assert ex == 900
        self.sets += 1
        if self.failure == "write":
            raise RedisError("Private result must not appear in diagnostics.")
        self.values[key] = value


def request_tools(
    database: Database,
    redis: Redis,
    *,
    user: str = "user",
    repository: str = "repo",
    rerank: bool = False,
) -> tuple[ToolRegistry, CodeTools]:
    registry = ToolRegistry(
        RequestScope(
            user_id=user,
            repository_id=repository,
            capabilities=frozenset({"code.read"}),
        )
    )
    return registry, CodeTools(
        registry,
        None,
        database.sha,
        pool=Pool(database),
        redis=redis,
        last_indexed_at=database.revision,
        rerank=rerank,
    )


def test_cross_request_hits_keep_request_labels_and_first_pass(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.observability")
    database, redis = Database(), Redis()
    queries = scripted_retriever(monkeypatch, database)
    creations = []
    monkeypatch.setattr(
        module, "create_embedding_model", lambda: creations.append(True) or object()
    )

    async def run() -> None:
        registry, first = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0].ainvoke({"query": "initial"})
            await tools[1].ainvoke({"path": "app.py", "start_line": 3, "end_line": 4})
        registry, second = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            file = json.loads(
                await tools[1].ainvoke(
                    {"path": "app.py", "start_line": 3, "end_line": 4}
                )
            )
            search = json.loads(await tools[0].ainvoke({"query": "initial"}))
        assert file["sources"][0]["source_index"] == 1
        assert search["sources"][0]["source_index"] == 2
        assert first.first_pass[0].id == second.first_pass[0].id
        assert second.first_pass[0].source_index == 2
        assert second.first_query == "initial" and second.embeddings is None
        assert second.cache_stats["hits"] == 2
        assert redis.sets == 2  # Hits do not rewrite/refresh entries.
        assert all("source_index" not in raw.decode() for raw in redis.values.values())
        assert sum('FROM "RepositoryFile"' in sql for sql, *_ in database.queries) == 1
        assert database.active == 0

    asyncio.run(run())
    assert queries == ["initial"] and creations == [True]
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "app.observability"
    ]
    assert (
        sum(event["stage"] == "redis" and event["status"] == "hit" for event in events)
        == 2
    )
    assert (
        sum(event["stage"] == "redis" and event["status"] == "miss" for event in events)
        == 2
    )
    assert any(
        event["stage"] == "database" and event["operation"] == "file_context"
        for event in events
    )
    assert "app.py" not in caplog.text and "initial" not in caplog.text


@pytest.mark.parametrize("explicit", [False, True])
def test_cached_context_notes_preserve_required_read_gaps(explicit: bool) -> None:
    database, redis = Database(), Redis()
    database.files["large.py"] = "x" * 8000 + "\n" + "y" * 8000 + "\ntail"

    async def run() -> None:
        results = []
        for _ in range(2):
            registry, code = request_tools(database, redis)
            async with registry.scoped_tools(frozenset({"code.read"})) as tools:
                arguments: dict[str, Any] = {"path": "large.py"}
                if explicit:
                    arguments["end_line"] = 2
                results.append(json.loads(await tools[1].ainvoke(arguments)))
        assert results[0] == results[1]
        assert bool(results[1]["missing_information"]) is explicit
        assert bool(results[1]["context_notes"]) is not explicit
        assert results[1]["file_line_count"] == 3
        assert code.cache_stats["hits"] == 1
        assert all(key.startswith("repomind:code-context:v4:") for key in redis.values)
        assert sum('FROM "RepositoryFile"' in sql for sql, *_ in database.queries) == 1

    asyncio.run(run())


@pytest.mark.parametrize("old_version", ["v2", "v3"])
@pytest.mark.parametrize("search", [False, True])
def test_context_cache_v4_does_not_consume_old_namespace_entries(
    monkeypatch: pytest.MonkeyPatch,
    old_version: str,
    search: bool,
) -> None:
    database, redis = Database(), Redis()
    queries = scripted_retriever(monkeypatch, database)
    monkeypatch.setattr(module, "create_embedding_model", lambda: object())
    arguments = {"query": "initial"} if search else {"path": "app.py"}

    async def run() -> None:
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0 if search else 1].ainvoke(arguments)
        key, raw = next(iter(redis.values.items()))
        old_key = key.replace(":v4:", f":{old_version}:")
        redis.values[old_key] = raw.replace(key.encode(), old_key.encode())
        del redis.values[key]
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0 if search else 1].ainvoke(arguments)
        assert code.cache_stats["hits"] == 0
        assert old_key in redis.values and key in redis.values
        registry, warm = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[0 if search else 1].ainvoke(arguments)
        assert warm.cache_stats["hits"] == 1 and redis.sets == 2
        if search:
            assert queries == ["initial", "initial"]
        else:
            assert (
                sum('FROM "RepositoryFile"' in sql for sql, *_ in database.queries) == 2
            )

    asyncio.run(run())


def test_cache_cannot_bypass_revocation_or_repository_isolation() -> None:
    database, redis = Database(), Redis()

    async def run() -> None:
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[1].ainvoke({"path": "app.py"})
            gets = redis.gets
            database.allowed = False
            with pytest.raises(Exception, match="not authorized"):
                await tools[1].ainvoke({"path": "app.py"})
            assert redis.gets == gets
        database.allowed = True
        registry, other = request_tools(database, redis, repository="other")
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            with pytest.raises(Exception, match="not authorized"):
                await tools[1].ainvoke({"path": "app.py"})
        assert redis.gets == gets
        database.users.add("second")
        registry, second = request_tools(database, redis, user="second")
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            assert json.loads(await tools[1].ainvoke({"path": "app.py"}))["sources"]
        assert second.cache_stats["hits"] == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "change", ["commit", "rebuild", "settings", "expiry", "corrupt", "oversized"]
)
def test_changed_index_configuration_and_expired_or_corrupt_entries_miss(
    change: str,
) -> None:
    database, redis = Database(), Redis()

    async def run() -> None:
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[1].ainvoke({"path": "app.py"})
            if change in {"commit", "rebuild"}:
                if change == "commit":
                    database.sha = "b" * 40
                else:
                    database.revision += timedelta(seconds=1)
                with pytest.raises(Exception, match="snapshot changed"):
                    await tools[1].ainvoke({"path": "app.py"})
            elif change == "expiry":
                redis.values.clear()
            elif change == "corrupt":
                for key in redis.values:
                    redis.values[key] = b'{"malformed": true}'
            elif change == "oversized":
                for key in redis.values:
                    redis.values[key] = b"x" * (module.CACHE_MAX_BYTES + 1)
        registry, current = request_tools(database, redis, rerank=change == "settings")
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            assert json.loads(await tools[1].ainvoke({"path": "app.py"}))["sources"]
        assert current.cache_stats["hits"] == 0
        assert sum('FROM "RepositoryFile"' in sql for sql, *_ in database.queries) == 2

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["read", "write", "timeout"])
def test_cache_outage_and_deadline_preserve_authorized_reads(
    monkeypatch: pytest.MonkeyPatch, caplog: Any, fault: str
) -> None:
    database, redis = Database(), Redis()
    redis.failure = fault
    monkeypatch.setattr(module, "CACHE_TIMEOUT_SECONDS", 0.001)

    async def run() -> None:
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            for _ in range(2):
                assert json.loads(await tools[1].ainvoke({"path": "app.py"}))["sources"]
        assert code.cache_stats["errors"] == 1
        assert database.active == 0

    asyncio.run(run())
    assert "Private" not in caplog.text
    assert redis.canceled == (fault == "timeout")


def test_zero_ttl_disables_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_CONTEXT_CACHE_TTL_SECONDS", "0")
    get_settings.cache_clear()
    database, redis = Database(), Redis()

    async def run() -> None:
        registry, code = request_tools(database, redis)
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            await tools[1].ainvoke({"path": "app.py"})
        assert redis.gets == redis.sets == 0

    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()


def test_cache_cancellation_releases_read_and_propagates() -> None:
    database = Database()

    async def run() -> None:
        started = asyncio.Event()

        class BlockedRedis(Redis):
            async def get(self, key: str) -> bytes | None:
                started.set()
                await asyncio.Event().wait()
                return None

        registry, code = request_tools(database, BlockedRedis())
        async with registry.scoped_tools(frozenset({"code.read"})) as tools:
            task = asyncio.create_task(tools[1].ainvoke({"path": "app.py"}))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert database.active == 0
        assert code.cache_stats["errors"] == 0

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_concurrent_cached_reads_keep_exact_paths_and_stable_ids() -> None:
    database, redis = Database(), Redis()
    database.files[" app.py "] = database.files["app.py"]

    async def run() -> None:
        for warm in (False, True):
            registry, code = request_tools(database, redis)
            async with registry.scoped_tools(frozenset({"code.read"})) as tools:
                results = await asyncio.gather(
                    *(tools[1].ainvoke({"path": " app.py "}) for _ in range(2))
                )
            sources = [json.loads(result)["sources"][0] for result in results]
            assert sources[0] == sources[1]
            assert sources[0]["path"] == " app.py "
            assert "%20app.py%20" in sources[0]["url"]
            assert len(code.evidence) == 1 and database.active == 0
            if warm:
                assert code.cache_stats["hits"] == 2
        assert database.maximum > 1

    asyncio.run(run())
