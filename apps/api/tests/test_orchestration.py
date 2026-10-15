"""Offline router, history, and SSE regressions using the real graph and client."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn
from uuid import UUID

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import JsonValue, TypeAdapter, ValidationError

from app import chat, orchestration
from app.config import get_settings
from app.evals.models import Dataset, Question
from app.orchestration import RouteDecision, create_query_router

Provider = tuple[list[dict[str, JsonValue]], list[str | httpx.HTTPError]]


@pytest.fixture
def provider(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Provider]:
    """Capture real client requests and supply offline provider responses."""
    requests: list[dict[str, JsonValue]] = []
    replies: list[str | httpx.HTTPError] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(
            TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        )
        assert request.headers["authorization"] == "Bearer test-hive-key"
        assert request.url == "https://api.thehive.ai/api/v3/chat/completions"
        reply = replies.pop(0)
        if isinstance(reply, httpx.HTTPError):
            raise reply
        return httpx.Response(
            200,
            json={
                "id": "offline-router",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    def create_model(**kwargs: object) -> ChatOpenAI:
        return ChatOpenAI(http_async_client=client, **kwargs)

    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    monkeypatch.setenv("HIVE_CHAT_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    monkeypatch.setattr(chat, "ChatOpenAI", create_model)
    get_settings.cache_clear()
    try:
        yield requests, replies
    finally:
        asyncio.run(client.aclose())
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "scope,strategy",
    [
        ("CODE", None),
        ("GITHUB", None),
        ("MIXED", "PARALLEL"),
        ("MIXED", "GITHUB_THEN_CODE"),
        (None, None),
    ],
)
def test_router_uses_validated_json_without_research(
    scope: str | None,
    strategy: str | None,
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, replies = provider
    clarification = "Which problem and change do you mean?" if scope is None else None
    replies.append(
        json.dumps(
            {"scope": scope, "clarification": clarification, "mixed_strategy": strategy}
        )
    )

    def unexpected_research(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("Routing must not start retrieval, MCP, or answer generation.")

    for name in (
        "create_embedding_model",
        "RepoMindParentHybridRetriever",
        "scoped_mcp_tools",
        "stream_repository_answer",
    ):
        monkeypatch.setattr(chat, name, unexpected_research)

    router = create_query_router(chat.create_chat_model(purpose="router"))
    history = [HumanMessage(content="Read issue #101."), AIMessage(content="A bug.")]
    result = asyncio.run(
        router.ainvoke({"question": "What do its comments say?", "history": history})
    )
    assert result == {
        "decision": RouteDecision(
            scope=scope, clarification=clarification, mixed_strategy=strategy
        )
    }
    assert len(requests) == 1
    request = requests[0]
    assert request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 512
    assert request["stream"] is False
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0
    assert "max_thinking_tokens" not in request
    assert "tools" not in request
    messages = request["messages"]
    assert isinstance(messages, list)
    assert messages[1:] == [
        {"role": "user", "content": "Read issue #101."},
        {"role": "assistant", "content": "A bug."},
        {"role": "user", "content": "What do its comments say?"},
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "not JSON",
        '{"scope":"OTHER","clarification":null}',
        '{"scope":"CODE","clarification":"Which file?"}',
        '{"scope":null,"clarification":null}',
        '{"scope":null,"clarification":"   "}',
        '{"scope":"CODE","clarification":null,"extra":true}',
        '{"scope":"CODE"}',
        '{"scope":null,"clarification":42}',
        '{"scope":"MIXED","clarification":null}',
        '{"scope":"MIXED","clarification":null,"mixed_strategy":"OTHER"}',
        '{"scope":"CODE","clarification":null,"mixed_strategy":"PARALLEL"}',
        '{"scope":null,"clarification":"Which?","mixed_strategy":"GITHUB_THEN_CODE"}',
    ],
)
def test_invalid_decisions_are_rejected(reply: str) -> None:
    with pytest.raises(ValidationError):
        RouteDecision.model_validate_json(reply)


@pytest.mark.parametrize(
    "failure", ["invalid_json", "truncated_json", "invalid_decision", "transport"]
)
@pytest.mark.parametrize("recover", [False, True])
def test_router_retries_once_and_surfaces_safe_errors(
    failure: str,
    recover: bool,
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, replies = provider
    errors: dict[str, str | httpx.HTTPError] = {
        "invalid_json": "not JSON with private prompt content",
        "truncated_json": '{"scope":"CODE","clarification":null',
        "invalid_decision": '{"scope":null,"clarification":null}',
        "transport": httpx.ReadTimeout("private provider error"),
    }
    error = errors[failure]
    replies.extend(
        [error, '{"scope":"CODE","clarification":null}' if recover else error]
    )
    router = create_query_router(chat.create_chat_model(purpose="router"))
    if recover:
        result = asyncio.run(router.ainvoke({"question": "How?", "history": []}))
        assert result["decision"].scope == "CODE"
    else:
        with pytest.raises(
            RuntimeError, match="^Query routing failed after two"
        ) as exc:
            asyncio.run(router.ainvoke({"question": "How?", "history": []}))
        assert "private" not in str(exc.value)
        assert exc.value.__cause__ is not None
    assert len(requests) == 2


@pytest.mark.parametrize("recover", [False, True])
def test_router_deadline_cancels_each_attempt(
    recover: bool,
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, replies = provider
    calls: list[object] = []

    async def classify(
        messages: object, config: RunnableConfig
    ) -> dict[str, AIMessage]:
        calls.append(messages)
        if len(calls) == 1 or not recover:
            await asyncio.Event().wait()
        return {"raw": AIMessage(content='{"scope":"CODE","clarification":null}')}

    monkeypatch.setattr(
        ChatOpenAI,
        "with_structured_output",
        lambda *_args, **_kwargs: RunnableLambda(classify),
    )
    monkeypatch.setattr(orchestration, "ROUTER_TIMEOUT_SECONDS", 1)
    router = create_query_router(chat.create_chat_model(purpose="router"))
    if recover:
        assert (
            asyncio.run(router.ainvoke({"question": "How?", "history": []}))[
                "decision"
            ].scope
            == "CODE"
        )
    else:
        with pytest.raises(RuntimeError, match="^Query routing failed after two"):
            asyncio.run(router.ainvoke({"question": "How?", "history": []}))
    assert len(calls) == 2
    assert not requests and not replies


def test_cancellation_does_not_retry(
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    started = asyncio.Event()

    async def classify(messages: object, config: RunnableConfig) -> RouteDecision:
        calls.append(messages)
        started.set()
        await asyncio.Event().wait()
        return RouteDecision(scope="CODE", clarification=None)

    monkeypatch.setattr(
        ChatOpenAI,
        "with_structured_output",
        lambda *_args, **_kwargs: RunnableLambda(classify),
    )
    router = create_query_router(chat.create_chat_model(purpose="router"))

    async def cancel() -> None:
        task = asyncio.create_task(router.ainvoke({"question": "How?", "history": []}))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert len(calls) == 1


def test_graph_hides_router_messages_and_propagates_callbacks(
    provider: Provider,
) -> None:
    requests, replies = provider
    replies.append('{"scope":"CODE","clarification":null}')
    tags_seen: list[list[str]] = []

    class Callback(BaseCallbackHandler):
        def on_chat_model_start(
            self,
            serialized: dict[str, object],
            messages: list[list[object]],
            *,
            run_id: UUID,
            tags: list[str] | None = None,
            **kwargs: object,
        ) -> None:
            tags_seen.append(tags or [])

    router = create_query_router(chat.create_chat_model(purpose="router"))

    async def collect() -> list[object]:
        parts = []
        async for part in router.astream(
            {"question": "How is the GitHub tool implemented?", "history": []},
            {"callbacks": [Callback()], "tags": ["router-test"]},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            assert part["type"] != "messages"
            parts.append(part)
        return parts

    assert asyncio.run(collect()) == [
        {
            "type": "updates",
            "ns": (),
            "data": {
                "query_router": {
                    "decision": RouteDecision(scope="CODE", clarification=None)
                }
            },
        }
    ]
    assert len(requests) == 1
    assert tags_seen and all(
        "nostream" in tags and "router-test" in tags for tags in tags_seen
    )


@pytest.mark.parametrize(
    "question",
    Dataset.model_validate_json(
        (Path(__file__).parents[1] / "app/evals/data/dataset.json").read_text(
            encoding="utf-8"
        )
    ).questions,
    ids=lambda question: question.id,
)
def test_benchmark_questions_use_question_only_contract(
    question: Question,
    provider: Provider,
) -> None:
    requests, replies = provider
    replies.append(
        json.dumps(
            {
                "scope": question.evidence_scope,
                "clarification": None,
                "mixed_strategy": "PARALLEL"
                if question.evidence_scope == "MIXED"
                else None,
            }
        )
    )
    router = create_query_router(chat.create_chat_model(purpose="router"))
    result = asyncio.run(router.ainvoke({"question": question.question, "history": []}))
    assert result["decision"].scope == question.evidence_scope
    messages = requests[0]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[-1] == {"role": "user", "content": question.question}


@pytest.mark.parametrize("failure", [False, True])
def test_router_clarification_and_failure_preserve_sse_and_persistence(
    failure: bool,
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, replies = provider
    clarification = "Which PR and problem do you mean?"
    replies.extend(
        ["malformed private output"] * 2
        if failure
        else [json.dumps({"scope": None, "clarification": clarification})]
    )
    writes: list[tuple[object, ...]] = []

    class Connection:
        async def fetchrow(self, sql: str, *_args: object) -> dict[str, object]:
            if 'FROM "Repository"' in sql:
                return {
                    "id": "repo",
                    "htmlUrl": "https://github.com/fixture/repo",
                    "indexedCommitSha": "a" * 40,
                    "defaultBranch": "main",
                    "indexStatus": "READY",
                    "indexedChunkCount": 1,
                }
            return {"content": "Does it fix that?"}

        async def fetch(self, *_args: object) -> list[object]:
            return []

        async def execute(self, *args: object) -> str:
            writes.append(args)
            return "OK"

    def unexpected_research(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("Clarification and router failure must not start research.")

    for name in (
        "create_embedding_model",
        "RepoMindParentHybridRetriever",
        "scoped_mcp_tools",
        "stream_repository_answer",
    ):
        monkeypatch.setattr(chat, name, unexpected_research)

    router = create_query_router(chat.create_chat_model(purpose="router"))

    async def collect() -> list[str]:
        return [
            event
            async for event in chat.stream_repository_chat(
                Connection(),
                "repo",
                "conversation",
                "user",
                "message",
                query_router=router,
            )
        ]

    events = asyncio.run(collect())
    names = [event.splitlines()[0] for event in events]
    assert names[0] == "event: assistant_message"
    assert all('"scope"' not in event and "private" not in event for event in events)
    assert not any(
        'INSERT INTO "ChatMessageCitation"' in str(write[0]) for write in writes
    )
    if failure:
        assert names == ["event: assistant_message", "event: error"]
        assert "Query routing failed after two attempts" in events[-1]
        assert "'FAILED'" in str(writes[-1][0])
        assert len(requests) == 2
    else:
        assert names == [
            "event: assistant_message",
            "event: delta",
            "event: citations",
            "event: done",
        ]
        assert json.dumps({"text": clarification}) in events[1]
        assert '"citations": []' in events[2]
        assert any(
            "'SUCCEEDED'" in str(write[0]) and clarification in write
            for write in writes
        )
        assert len(requests) == 1


def test_latest_history_excludes_current_failed_and_other_conversations() -> None:
    with sqlite3.connect(":memory:") as database:
        database.row_factory = sqlite3.Row
        database.execute(
            'CREATE TABLE "ChatMessage" ("id" TEXT, "conversationId" TEXT, '
            '"role" TEXT, "content" TEXT, "status" TEXT, "createdAt" INTEGER)'
        )
        rows = [
            (
                f"m{index:02}",
                "conversation",
                "USER" if index % 2 == 0 else "ASSISTANT",
                f"message {index}",
                "SUCCEEDED",
                index,
            )
            for index in range(25)
        ]
        rows.extend(
            [
                ("current", "conversation", "USER", "current", "SUCCEEDED", 30),
                ("failed", "conversation", "ASSISTANT", "failed", "FAILED", 31),
                ("other", "other", "USER", "other", "SUCCEEDED", 32),
            ]
        )
        database.executemany(
            'INSERT INTO "ChatMessage" VALUES (?, ?, ?, ?, ?, ?)', rows
        )

        class Connection:
            async def fetch(self, query: str, *args: str) -> list[sqlite3.Row]:
                return database.execute(query, args).fetchall()

        history = asyncio.run(
            chat.load_history(Connection(), "conversation", "current")
        )
    assert [message.content for message in history] == [
        f"message {index}" for index in range(5, 25)
    ]
    assert all(
        isinstance(message, HumanMessage if index % 2 else AIMessage)
        for index, message in enumerate(history)
    )


def test_answer_stream_filters_internal_classification_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        async def astream(self, *_args: object, **_kwargs: object):
            for metadata in (
                {"langgraph_node": "query_router"},
                {"langgraph_node": "model", "tags": ["nostream"]},
                {"langgraph_node": "model", "tags": ["answer"]},
            ):
                yield {
                    "type": "messages",
                    "data": (AIMessageChunk(content="text"), metadata),
                }

    monkeypatch.setattr(chat, "create_agent", lambda *_args, **_kwargs: Agent())
    monkeypatch.setattr(chat, "create_chat_model", lambda: object())

    async def collect() -> list[tuple[str, str]]:
        return [event async for event in chat.stream_repository_answer("q", [], [], [])]

    assert asyncio.run(collect()) == [("delta", "text")]
