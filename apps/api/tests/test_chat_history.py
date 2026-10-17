"""Conversation cache regressions without model, database or Redis services."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import ConnectionError as RedisConnectionError
from test_chat_streaming import ChatDatabase, ChatPool

from app import chat
from app.config import get_settings
from app.tool_registry import RequestScope

SCOPE = RequestScope(
    user_id="user", repository_id="repo", capabilities=frozenset({"code.read"})
)


class HistoryConnection:
    def __init__(self) -> None:
        self.rows = [
            {
                "id": f"m{index:02}",
                "conversationId": "conversation",
                "role": "USER" if index % 2 == 0 else "ASSISTANT",
                "content": f"message {index}",
                "status": "SUCCEEDED",
                "createdAt": datetime(2026, 10, 1) + timedelta(seconds=index),
                "updatedAt": datetime(2026, 10, 1) + timedelta(seconds=index),
            }
            for index in range(26)
        ]
        self.reads = 0

    def window(self, conversation: str, current: str | None) -> list[dict[str, Any]]:
        return sorted(
            (
                row
                for row in self.rows
                if row["conversationId"] == conversation
                and row["id"] != current
                and row["status"] == "SUCCEEDED"
            ),
            key=lambda row: (row["createdAt"], row["id"]),
            reverse=True,
        )[:20]

    def revision(self, current: str | None = "m25") -> list[tuple[str, datetime]]:
        return [
            (row["id"], row["updatedAt"])
            for row in self.window("conversation", current)
        ]

    async def fetch(self, query: str, conversation: str, current: str | None) -> Any:
        self.reads += 1
        return self.window(conversation, current)


class HistoryRedis:
    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, int]] = []
        self.fault: str | None = None
        self.started = asyncio.Event()

    async def get(self, key: str) -> bytes | None:
        self.gets.append(key)
        self.started.set()
        if self.fault == "read":
            raise RedisConnectionError("private cache failure detail")
        if self.fault == "timeout":
            await asyncio.Event().wait()
        return self.entries.get(key)

    async def set(self, key: str, value: bytes, *, ex: int) -> None:
        self.sets.append((key, ex))
        if self.fault == "write":
            raise RedisConnectionError("private cache failure detail")
        if self.fault == "write_timeout":
            await asyncio.Event().wait()
        self.entries[key] = value


def test_history_hit_preserves_window_roles_order_and_exclusions() -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        connection.rows[24]["status"] = "FAILED"
        connection.rows[23]["status"] = "STREAMING"
        connection.rows[22]["conversationId"] = "other"
        expected = [
            row["content"] for row in reversed(connection.window("conversation", "m25"))
        ]
        for _ in range(2):
            history = await chat.load_history(
                connection,
                "conversation",
                "m25",
                redis=redis,
                scope=SCOPE,
                revision=connection.revision(),
            )
            assert [message.content for message in history] == expected
            assert len(history) == 20
            assert all(
                isinstance(message, (HumanMessage, AIMessage)) for message in history
            )
            assert connection.reads == 1
        assert len(redis.gets) == 2 and len(redis.sets) == 1
        assert redis.sets[0][1] == chat.HISTORY_CACHE_TTL_SECONDS
        assert set(json.loads(next(iter(redis.entries.values())))) == {
            "identity",
            "messages",
        }

    asyncio.run(run())


@pytest.mark.parametrize("change", ["late_completion", "edit", "delete", "new_user"])
def test_changed_window_cannot_reuse_old_history(change: str) -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        if change == "late_completion":
            connection.rows[9]["status"] = "STREAMING"
        await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        old_key = redis.gets[-1]
        row = connection.rows[9]
        if change == "late_completion":
            row["status"] = "SUCCEEDED"
        elif change == "edit":
            row["content"] = "Corrected answer."
        elif change == "delete":
            connection.rows.remove(row)
        else:
            connection.rows[25]["id"] = "previous_unanswered_question"
        row["updatedAt"] += timedelta(hours=1)
        fresh = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert redis.gets[-1] != old_key and connection.reads == 2
        assert [message.content for message in fresh] == [
            row["content"] for row in reversed(connection.window("conversation", "m25"))
        ]
        # A delayed writer can republish the old immutable key without
        # overwriting the version selected by the next request.
        await redis.set(old_key, redis.entries[old_key], ex=300)
        again = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert again == fresh and connection.reads == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "different", ["user", "repository", "conversation", "database"]
)
def test_history_cache_isolation(
    monkeypatch: pytest.MonkeyPatch, different: str
) -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        scope = RequestScope(
            user_id="other" if different == "user" else SCOPE.user_id,
            repository_id="other" if different == "repository" else SCOPE.repository_id,
            capabilities=SCOPE.capabilities,
        )
        if different == "database":
            monkeypatch.setenv(
                "API_DATABASE_URL", "postgresql://cache-test@localhost/other"
            )
            get_settings.cache_clear()
        await chat.load_history(
            connection,
            "other" if different == "conversation" else "conversation",
            "m25",
            redis=redis,
            scope=scope,
            revision=connection.revision(),
        )
        assert redis.gets[0] != redis.gets[1] and connection.reads == 2

    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "malformed",
    [
        "json",
        "identity",
        "revision",
        "role",
        "extra",
        "oversized",
        "too_many",
        "current",
    ],
)
def test_invalid_cache_entries_fall_back_to_database(malformed: str) -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        key = redis.gets[-1]
        entry = json.loads(redis.entries[key])
        if malformed == "identity":
            entry["identity"] = "other"
        elif malformed == "revision":
            entry["messages"][0]["updated_at"] = "2026-01-01T00:00:00"
        elif malformed == "role":
            entry["messages"][0]["role"] = "SYSTEM"
        elif malformed == "extra":
            entry["reasoning"] = "Internal output must not become history."
        elif malformed == "too_many":
            entry["messages"] *= 2
        elif malformed == "current":
            entry["messages"][0]["id"] = "m25"
        redis.entries[key] = (
            b"{"
            if malformed == "json"
            else b"x" * (chat.HISTORY_CACHE_MAX_BYTES + 1)
            if malformed == "oversized"
            else json.dumps(entry).encode()
        )
        history = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert connection.reads == 2 and len(history) == 20
        assert [message.content for message in history] == [
            f"message {i}" for i in range(5, 25)
        ]

    asyncio.run(run())


def test_overlapping_refreshes_cannot_replace_newer_history() -> None:
    async def run() -> None:
        connection = HistoryConnection()
        connection.rows[9]["status"] = "STREAMING"
        blocked, release = asyncio.Event(), asyncio.Event()

        class DelayedRedis(HistoryRedis):
            async def set(self, key: str, value: bytes, *, ex: int) -> None:
                if not blocked.is_set():
                    blocked.set()
                    await release.wait()
                await super().set(key, value, ex=ex)

        redis = DelayedRedis()
        older = asyncio.create_task(
            chat.load_history(
                connection,
                "conversation",
                "m25",
                redis=redis,
                scope=SCOPE,
                revision=connection.revision(),
            )
        )
        await blocked.wait()
        connection.rows[9]["status"] = "SUCCEEDED"
        connection.rows[9]["updatedAt"] += timedelta(hours=1)
        newer = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        release.set()
        await older
        cached = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert cached == newer and connection.reads == 2
        assert "message 9" in [message.content for message in cached]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("fault", ["read", "write", "timeout", "write_timeout"])
def test_cache_failures_are_bounded_and_keep_history(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fault: str
) -> None:
    monkeypatch.setattr(chat, "HISTORY_CACHE_TIMEOUT_SECONDS", 0.01)
    caplog.set_level(logging.INFO, logger="app")

    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        redis.fault = fault
        history = await asyncio.wait_for(
            chat.load_history(
                connection,
                "conversation",
                "m25",
                redis=redis,
                scope=SCOPE,
                revision=connection.revision(),
                request_id="safe-request-id",
            ),
            timeout=1,
        )
        assert len(history) == 20 and connection.reads == 1

    asyncio.run(run())
    assert "private cache failure detail" not in caplog.text
    assert "message 24" not in caplog.text
    assert "safe-request-id" in caplog.text
    assert '"status": "failed"' in caplog.text


def test_cache_cancellation_propagates_without_database_fallback() -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        redis.fault = "timeout"
        task = asyncio.create_task(
            chat.load_history(
                connection,
                "conversation",
                "m25",
                redis=redis,
                scope=SCOPE,
                revision=connection.revision(),
            )
        )
        await redis.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert connection.reads == 0

    asyncio.run(run())


@pytest.mark.parametrize("oversized", [False, True])
def test_cache_expiry_or_oversized_history_keeps_database_fallback(
    oversized: bool,
) -> None:
    async def run() -> None:
        connection, redis = HistoryConnection(), HistoryRedis()
        if oversized:
            connection.rows[24]["content"] = "x" * chat.HISTORY_CACHE_MAX_BYTES
        first = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert bool(redis.entries) is not oversized
        redis.entries.clear()
        second = await chat.load_history(
            connection,
            "conversation",
            "m25",
            redis=redis,
            scope=SCOPE,
            revision=connection.revision(),
        )
        assert first == second and connection.reads == 2

    asyncio.run(run())


def test_no_redis_keeps_history_contract() -> None:
    connection = HistoryConnection()
    history = asyncio.run(chat.load_history(connection, "conversation", "m25"))
    assert len(history) == 20 and connection.reads == 1


@pytest.mark.parametrize("revoked", [False, True])
def test_refresh_failure_preserves_committed_answer_and_access_checks_precede_cache(
    monkeypatch: pytest.MonkeyPatch, revoked: bool
) -> None:
    monkeypatch.setenv("HIVE_API_KEY", "offline")
    get_settings.cache_clear()
    database, redis = ChatDatabase(), HistoryRedis()
    database.question = "hi"
    database.allowed = not revoked
    original = chat.load_history

    async def history(
        connection: Any, conversation: str, current: str | None, **kwargs: Any
    ) -> Any:
        if current is None:
            raise RuntimeError("Controlled refresh failure.")
        return await original(connection, conversation, current, **kwargs)

    monkeypatch.setattr(chat, "load_history", history)

    async def run() -> None:
        events = [
            event
            async for event in chat.stream_repository_chat(
                ChatPool(database),
                "repo",
                "conversation",
                "user",
                "message",
                redis=redis,
            )
        ]
        if revoked:
            assert not redis.gets and not database.messages
            assert events[-1].startswith("event: error")
        else:
            assert len(redis.gets) == 1
            assert events[-1].startswith("event: done")
            assert next(iter(database.messages.values()))["status"] == "SUCCEEDED"

    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()
