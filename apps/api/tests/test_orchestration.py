"""Offline router, history, and SSE regressions using the real graph and client."""

import asyncio
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn
from uuid import UUID

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
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
            {
                "scope": scope,
                "clarification": clarification,
                "mixed_strategy": strategy,
                "code_task": "Inspect indexed behavior." if scope == "MIXED" else None,
                "github_task": "Read requested live facts."
                if scope == "MIXED"
                else None,
            }
        )
    )

    def unexpected_research(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("Routing must not start retrieval, MCP, or answer generation.")

    monkeypatch.setattr("app.code_tools.create_embedding_model", unexpected_research)
    monkeypatch.setattr("app.tool_registry.MultiServerMCPClient", unexpected_research)

    router = create_query_router(chat.create_chat_model(purpose="router"))
    history = [HumanMessage(content="Read issue #101."), AIMessage(content="A bug.")]
    result = asyncio.run(
        router.ainvoke({"question": "What do its comments say?", "history": history})
    )
    assert result == {
        "decision": RouteDecision(
            scope=scope,
            clarification=clarification,
            mixed_strategy=strategy,
            code_task="Inspect indexed behavior." if scope == "MIXED" else None,
            github_task="Read requested live facts." if scope == "MIXED" else None,
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


def test_router_demonstrations_use_complete_valid_decisions(provider: Provider) -> None:
    requests, replies = provider
    replies.append('{"scope":"GITHUB","clarification":null}')
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": "Read PR #218.", "history": []}))
    prompt = requests[0]["messages"][0]["content"]
    examples = re.findall(r"^Output: (.+)$", prompt, flags=re.MULTILINE)
    assert examples
    for example in examples:
        RouteDecision.model_validate_json(example)
        assert set(json.loads(example)) == set(RouteDecision.model_fields)


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
        '{"scope":"MIXED","clarification":null,"mixed_strategy":"PARALLEL",'
        '"code_task":" ","github_task":"Read live facts."}',
        '{"scope":"CODE","clarification":null,"code_task":"Unexpected assignment."}',
        '{"scope":"GITHUB","clarification":null,'
        '"github_task":"Unexpected assignment."}',
        '{"scope":"MIXED","clarification":null,"mixed_strategy":"PARALLEL",'
        '"code_task":"Read code.","github_task":"Read issue.",'
        '"github_evidence":"OTHER"}',
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
                "code_task": "Inspect indexed behavior."
                if question.evidence_scope == "MIXED"
                else None,
                "github_task": "Read requested live facts."
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


@pytest.mark.parametrize("turn_count", [0, 1, 3, 8])
def test_router_keeps_recent_whole_turns_without_mutating_history(
    turn_count: int, provider: Provider
) -> None:
    requests, replies = provider
    replies.append('{"scope":"CODE","clarification":null}')
    history = [AIMessage(content="Orphaned answer at the database window boundary.")]
    for index in range(turn_count):
        history.extend(
            [
                HumanMessage(content=f"Explain indexed component {index}."),
                AIMessage(content=f"Component {index} handles input."),
            ]
        )
    original = [message.model_dump() for message in history]
    question = "Explain the indexed implementation."
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": question, "history": history}))
    expected = history[1:][-6:]
    assert requests[0]["messages"][1:] == [
        *[
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
            }
            for message in expected
        ],
        {"role": "user", "content": question},
    ]
    assert [message.model_dump() for message in history] == original
    assert len(requests) == 1


def test_router_retains_unanswered_user_turn_for_retry_references(
    provider: Provider,
) -> None:
    requests, replies = provider
    replies.append('{"scope":"GITHUB","clarification":null}')
    history = [
        HumanMessage(content="Explain the implementation."),
        AIMessage(content="Indexed implementation."),
        HumanMessage(content="Read issue #214."),
        HumanMessage(content="Read PR #215 instead."),
    ]
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": "Try that again.", "history": history}))
    assert requests[0]["messages"][1:-1] == [
        {"role": "user", "content": "Explain the implementation."},
        {"role": "assistant", "content": "Indexed implementation."},
        {"role": "user", "content": "Read issue #214."},
        {"role": "user", "content": "Read PR #215 instead."},
    ]


@pytest.mark.parametrize("newest_oversized", [False, True])
def test_router_budget_drops_whole_oldest_turns_not_partial_answers(
    newest_oversized: bool, provider: Provider
) -> None:
    requests, replies = provider
    replies.append('{"scope":null,"clarification":"Which subject do you mean?"}')
    small = [HumanMessage(content="Read issue #214."), AIMessage(content="A report.")]
    oversized = [
        HumanMessage(content="Explain architecture."),
        AIMessage(content="x" * 9000),
    ]
    history = [*small, *oversized] if newest_oversized else [*small, *oversized, *small]
    original = [message.model_dump() for message in history]
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": "What about it?", "history": history}))
    assert requests[0]["messages"][1:-1] == (
        []
        if newest_oversized
        else [
            {"role": "user", "content": "Read issue #214."},
            {"role": "assistant", "content": "A report."},
        ]
    )
    assert [message.model_dump() for message in history] == original


def test_long_architecture_answer_does_not_displace_two_recent_comparison_subjects(
    provider: Provider,
) -> None:
    requests, replies = provider
    replies.append(
        json.dumps(
            {
                "scope": "MIXED",
                "clarification": None,
                "mixed_strategy": "PARALLEL",
                "code_task": "Read indexed revision.",
                "github_task": "Read live revision.",
            }
        )
    )
    history = [
        HumanMessage(content="Explain the architecture."),
        AIMessage(content="Component documentation. " * 600),
        HumanMessage(content="Which revision is indexed?"),
        AIMessage(content="Indexed revision: " + "a" * 40),
        HumanMessage(content="What is the live revision?"),
        AIMessage(content="Live revision: " + "a" * 40),
    ]
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": "Do those agree?", "history": history}))
    assert requests[0]["messages"][1:-1] == [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": message.content,
        }
        for message in history[-4:]
    ]


@pytest.mark.parametrize("fits", [False, True])
def test_router_budget_keeps_an_exactly_fitting_turn(
    fits: bool, provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, replies = provider
    replies.append('{"scope":"CODE","clarification":null}')
    history = [HumanMessage(content="Read parser.py."), AIMessage(content="A parser.")]
    monkeypatch.setattr(
        orchestration,
        "ROUTER_HISTORY_TOKEN_BUDGET",
        count_tokens_approximately(history) - (0 if fits else 1),
    )
    router = create_query_router(chat.create_chat_model(purpose="router"))
    asyncio.run(router.ainvoke({"question": "Explain parser.py.", "history": history}))
    assert len(requests[0]["messages"][1:-1]) == (2 if fits else 0)


@pytest.mark.parametrize(
    "question",
    [
        "hi",
        "HI!",
        "hi again",
        "hello",
        "hey there",
        " Hello, RepoMind! ",
        "good morning",
    ],
)
@pytest.mark.parametrize("with_history", [False, True])
def test_standalone_greeting_never_inherits_an_investigation(
    question: str,
    with_history: bool,
    provider: Provider,
) -> None:
    requests, _ = provider
    history = (
        [
            HumanMessage(content="Explain the purpose and architecture."),
            AIMessage(content="An indexed architecture explanation."),
            HumanMessage(content="Compare it with https://github.com/example/other."),
            AIMessage(content="The previous repository discussion."),
            HumanMessage(content="hi"),
            AIMessage(
                content="An unrelated continuation of the previous investigation."
            ),
        ]
        if with_history
        else []
    )
    router = create_query_router(chat.create_chat_model(purpose="router"))
    result = asyncio.run(router.ainvoke({"question": question, "history": history}))
    assert result["decision"] == RouteDecision(
        scope=None,
        clarification="Hi! What would you like to know about this repository?",
    )
    assert requests == []


@pytest.mark.parametrize(
    "question,scope",
    [
        ("Hi, are WebSockets implemented?", "CODE"),
        ("Hello, what do its latest comments say?", "GITHUB"),
        ("Explain hi()", "CODE"),
        ("hi.py", "CODE"),
        ("hi --help", "CODE"),
    ],
)
def test_greeting_words_do_not_swallow_repository_questions(
    question: str,
    scope: str,
    provider: Provider,
) -> None:
    requests, replies = provider
    replies.append(json.dumps({"scope": scope, "clarification": None}))
    history = [HumanMessage(content="Read PR #73."), AIMessage(content="A live PR.")]
    router = create_query_router(chat.create_chat_model(purpose="router"))
    result = asyncio.run(router.ainvoke({"question": question, "history": history}))
    assert result["decision"].scope == scope
    assert len(requests) == 1
    assert requests[0]["messages"][-1] == {"role": "user", "content": question}
