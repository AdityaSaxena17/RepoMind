"""Native provider chunks through the specialist, parent graph and production SSE."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from test_code_tools import SHA, Connection, Database

from app import chat
from app.agents import PARTIAL_ANSWER_PREFIX
from app.config import get_settings


class ChatDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.messages: dict[str, dict[str, Any]] = {}
        self.citations: list[Any] = []
        self.fail_persistence = False
        self.revision = datetime(2026, 10, 15, tzinfo=timezone.utc)


class ChatConnection(Connection):
    @asynccontextmanager
    async def transaction(self, **kwargs: Any) -> Any:
        if kwargs:
            async with super().transaction(**kwargs):
                yield
            return
        messages = copy.deepcopy(self.database.messages)
        citations = list(self.database.citations)
        try:
            yield
        except BaseException:
            self.database.messages, self.database.citations = messages, citations
            raise

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        if 'FROM "ChatMessage"' in sql:
            return {"content": "Read the code."}
        if 'FROM "Repository"' in sql:
            return (
                {
                    "indexedCommitSha": SHA,
                    "lastIndexedAt": self.database.revision,
                    "indexStatus": "READY",
                    "indexedChunkCount": 1,
                    "htmlUrl": "https://github.com/example/project",
                }
                if self.database.allowed
                else None
            )
        return await super().fetchrow(sql, *args)

    async def fetch(self, *args: Any) -> Any:
        return []

    async def fetchval(self, *args: Any) -> Any:
        return self.database.allowed

    async def execute(self, sql: str, *args: Any) -> Any:
        if 'INSERT INTO "ChatMessageCitation"' in sql:
            if self.database.fail_persistence:
                raise RuntimeError("Controlled persistence failure.")
            self.database.citations.append(args)
        elif 'INSERT INTO "ChatMessage"' in sql:
            self.database.messages[args[0]] = {"status": "STREAMING", "content": ""}
        elif 'UPDATE "ChatMessage"' in sql:
            self.database.messages[args[0]].update(
                status="FAILED" if "'FAILED'" in sql else "SUCCEEDED", content=args[1]
            )
        return "OK"


class ChatPool:
    def __init__(self, database: ChatDatabase) -> None:
        self.database = database

    @asynccontextmanager
    async def acquire(self, **kwargs: Any) -> Any:
        connection = ChatConnection(self.database)
        yield connection
        assert not connection.in_transaction


@pytest.mark.parametrize("startup_failure", [False, True])
def test_application_lifespan_owns_pool_and_closes_redis_after_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    startup_failure: bool,
) -> None:
    from app import main

    events = []

    class Redis:
        async def aclose(self) -> None:
            events.append("redis_closed")

    class Pool:
        async def close(self) -> None:
            events.append("pool_closed")

    async def create_redis() -> Any:
        return Redis()

    async def create_pool(*args: Any, **kwargs: Any) -> Any:
        if startup_failure:
            raise RuntimeError("Pool startup failed.")
        return Pool()

    monkeypatch.setattr(main, "create_redis_pool", create_redis)
    monkeypatch.setattr(main.asyncpg, "create_pool", create_pool)

    async def run() -> None:
        async with main.application_lifespan(main.create_app()):
            events.append("started")

    if startup_failure:
        with pytest.raises(RuntimeError, match="Pool startup failed"):
            asyncio.run(run())
        assert events == ["redis_closed"]
    else:
        asyncio.run(run())
        assert events == ["started", "pool_closed", "redis_closed"]


class Wire(httpx.AsyncByteStream):
    def __init__(
        self, chunks: list[dict[str, Any]], pause: asyncio.Event | None
    ) -> None:
        self.chunks, self.pause = chunks, pause
        self.finished = False
        self.fail = False
        self.fail_start = False

    async def __aiter__(self) -> Any:
        for index, chunk in enumerate(self.chunks):
            yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            if self.fail_start and index == 0:
                raise httpx.RemoteProtocolError("Controlled empty-start disconnect.")
            if self.pause is not None and index == 3:
                await self.pause.wait()
                if self.fail:
                    raise httpx.ReadTimeout("Controlled mid-generation failure.")
        self.finished = True
        yield b"data: [DONE]\n\n"


def wire_chunks(
    deltas: list[tuple[dict[str, Any], str | None]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "offline",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        for delta, finish in deltas
    ]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "citation",
        "persistence",
        "late_tool",
        "unmarked",
        "revoked",
        "transport",
        "heartbeat",
        "whitespace",
        "closing_marker",
        "truncated",
        "missing_stop",
        "disconnect",
        "partial",
        "corrected_range",
        "last_turn",
        "unmarked_no_budget",
        "unmarked_repair_fails",
        "initial_transport",
        "initial_transport_repeat",
    ],
)
def test_direct_specialist_streams_native_final_tokens_without_extra_call(
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    monkeypatch.setenv("HIVE_API_KEY", "offline")
    get_settings.cache_clear()
    if fault == "heartbeat":
        monkeypatch.setattr(chat, "HEARTBEAT_SECONDS", 0.01)
    if fault in {"last_turn", "unmarked_no_budget"}:
        monkeypatch.setenv("AGENT_MODEL_CALL_LIMIT", "2")
        get_settings.cache_clear()
    database = ChatDatabase()
    database.fail_persistence = fault == "persistence"
    requests: list[Any] = []
    wires: list[Wire] = []

    async def run() -> None:
        pause = asyncio.Event()

        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if not payload.get("stream"):
                return httpx.Response(
                    200,
                    json={
                        "id": "router",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "offline",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {"scope": "CODE", "clarification": None}
                                    ),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read-1",
                        "type": "function",
                        "function": {
                            "name": "get_file_context",
                            "arguments": '{"path":"app.py"}',
                        },
                    }
                ]
            }
            omit_marker = fault in {
                "unmarked",
                "unmarked_no_budget",
                "unmarked_repair_fails",
            } and (len(requests) == 3 or fault == "unmarked_repair_fails")
            if not any(
                message.get("role") == "tool" for message in payload["messages"]
            ) and not (
                fault in {"unmarked", "unmarked_repair_fails"} and len(requests) == 4
            ):
                if fault == "corrected_range":
                    call["tool_calls"][0]["function"]["arguments"] = (
                        '{"path":"app.py","end_line":300}'
                    )
                deltas = [
                    (
                        {"role": "assistant", "content": "Private intermediate text"},
                        None,
                    ),
                    (call, None),
                    ({}, "tool_calls"),
                ]
                wire = Wire(wire_chunks(deltas), None)
                wire.fail_start = fault in {
                    "initial_transport",
                    "initial_transport_repeat",
                } and (len(requests) == 2 or fault == "initial_transport_repeat")
            elif (
                fault in {"partial", "corrected_range"}
                and sum(m.get("role") == "tool" for m in payload["messages"]) == 1
            ):
                call["tool_calls"][0]["id"] = "read-2"
                call["tool_calls"][0]["function"]["arguments"] = (
                    '{"path":"missing.py"}'
                    if fault == "partial"
                    else '{"path":"app.py"}'
                )
                if fault == "corrected_range":
                    feedback = next(
                        m["content"]
                        for m in payload["messages"]
                        if m.get("role") == "tool"
                    )
                    assert (
                        "250" in feedback
                        and "Split" in feedback
                        and len(feedback) < 500
                    )
                wire = Wire(wire_chunks([(call, None), ({}, "tool_calls")]), None)
            else:
                deltas = [
                    (
                        {"role": "assistant", "reasoning_content": "Private reasoning"},
                        None,
                    ),
                    ({"content": "<fi" if not omit_marker else "Unmarked "}, None),
                    (
                        {"content": "nal_answer>" if not omit_marker else "response "},
                        None,
                    ),
                    (
                        {"content": "\nValue " if fault == "whitespace" else "Value "},
                        None,
                    ),
                    ({"content": "is one "}, None),
                    (
                        {
                            "content": "[999]."
                            if fault == "citation"
                            else "[1].\n"
                            if fault == "whitespace"
                            else "[1]."
                        },
                        None,
                    ),
                ]
                if fault == "closing_marker":
                    deltas.extend(
                        [
                            ({"content": "</final_"}, None),
                            ({"content": "answer>"}, None),
                            ({"content": "\n"}, None),
                        ]
                    )
                if fault == "late_tool":
                    call["tool_calls"][0]["id"] = "forbidden-2"
                    deltas.extend([(call, None), ({}, "tool_calls")])
                else:
                    deltas.append(
                        (
                            {},
                            "length"
                            if fault == "truncated"
                            else None
                            if fault == "missing_stop"
                            else "stop",
                        )
                    )
                wire = Wire(wire_chunks(deltas), pause if not omit_marker else None)
                wire.fail = fault == "transport"
            wires.append(wire)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=wire
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            native_model = chat.ChatOpenAI
            monkeypatch.setattr(
                chat,
                "ChatOpenAI",
                lambda **kwargs: native_model(http_async_client=client, **kwargs),
            )
            events = []
            saw_live = False
            stream = chat.stream_repository_chat(
                ChatPool(database), "repo", "conversation", "user", "message"
            )
            if fault == "disconnect":
                from starlette.requests import ClientDisconnect

                from app.routers.chat import ChatStreamingResponse

                async def send(message: Any) -> None:
                    nonlocal saw_live
                    if message["type"] == "http.response.body":
                        event = message["body"].decode()
                        events.append(event)
                        if event.startswith("event: delta"):
                            assert not wires[-1].finished
                            saw_live = True
                            raise OSError("Client disconnected during generation.")

                async def receive() -> Any:
                    pytest.fail("ASGI 2.4 must detect disconnect through send.")

                response = ChatStreamingResponse(stream, headers={})
                with pytest.raises(ClientDisconnect):
                    await response(
                        {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                    )
                assert not pause.is_set() and not wires[-1].finished
            else:
                async for event in stream:
                    events.append(event)
                    if event.startswith("event: delta") and not saw_live:
                        assert not wires[-1].finished and not pause.is_set()
                        saw_live = True
                        if fault == "revoked":
                            database.allowed = False
                        if fault != "heartbeat":
                            pause.set()
                    if event.startswith(": heartbeat"):
                        assert fault == "heartbeat"
                        if saw_live:
                            pause.set()
            text = "".join(
                json.loads(event.split("data: ")[1])["text"]
                for event in events
                if event.startswith("event: delta")
            )
            assert (
                "Private" not in text
                and "final_answer" not in text
                and "scope" not in text
            )
            assert saw_live == (
                fault
                not in {
                    "unmarked_no_budget",
                    "unmarked_repair_fails",
                    "initial_transport_repeat",
                }
            )
            assert len(requests) == (
                4
                if fault
                in {
                    "partial",
                    "corrected_range",
                    "unmarked",
                    "unmarked_repair_fails",
                    "initial_transport",
                }
                else 3
            )  # One router + tool request + specialist final generation.
            assert sum(
                'FROM "RepositoryFile"' in entry[0] for entry in database.queries
            ) == (
                2
                if fault == "partial"
                else 0
                if fault == "initial_transport_repeat"
                else 1
            )
            persisted = next(iter(database.messages.values()))
            assert persisted["content"] == text
            if fault == "last_turn":
                assert "tools" in requests[-2] and not requests[-1].get("tools")
            if fault not in {
                None,
                "heartbeat",
                "whitespace",
                "closing_marker",
                "partial",
                "corrected_range",
                "last_turn",
                "unmarked",
                "initial_transport",
            }:
                assert persisted["status"] == "FAILED"
                if fault != "disconnect":
                    assert events[-1].startswith("event: error")
                assert not database.citations
                assert not any(
                    event.startswith(("event: done", "event: citations"))
                    for event in events
                )
            else:
                assert text == (
                    PARTIAL_ANSWER_PREFIX + "Value is one [1]."
                    if fault == "partial"
                    else "\nValue is one [1].\n"
                    if fault == "whitespace"
                    else "Value is one [1]."
                )
                assert persisted["status"] == "SUCCEEDED"
                assert events[-1].startswith("event: done")
                assert len(database.citations) == 1
            assert database.active == 0

    try:
        asyncio.run(asyncio.wait_for(run(), timeout=10))
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "scope,strategy,repair",
    [
        ("GITHUB", None, False),
        ("GITHUB", None, True),
        ("MIXED", "PARALLEL", False),
        ("MIXED", "GITHUB_THEN_CODE", False),
    ],
)
def test_native_github_and_mixed_stream_only_the_final_generation(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    strategy: str | None,
    repair: bool,
) -> None:
    from test_agents import github_scope
    from test_tool_registry import provider

    monkeypatch.setenv("HIVE_API_KEY", "offline")
    monkeypatch.setenv("SYNTHESIS_REASONING_EFFORT", "low")
    get_settings.cache_clear()
    lifecycle = github_scope(monkeypatch)
    monkeypatch.setattr(chat, "create_github_provider", lambda *args: provider())
    database = ChatDatabase()
    url = "https://github.com/example/project/issues/1"
    requests, wires = [], []

    async def run() -> None:
        pause = asyncio.Event()

        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if not payload.get("stream"):
                return httpx.Response(
                    200,
                    json={
                        "id": "router",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "offline",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "scope": scope,
                                            "clarification": None,
                                            "mixed_strategy": strategy,
                                            "code_task": "Read indexed app.py."
                                            if scope == "MIXED"
                                            else None,
                                            "github_task": "Read live issue #1."
                                            if scope == "MIXED"
                                            else None,
                                        }
                                    ),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            tools = payload.get("tools", [])
            assert payload["reasoning_effort"] == (
                "low" if scope == "MIXED" and not tools else "high"
            )
            if tools:
                code = any(t["function"]["name"] == "get_file_context" for t in tools)
                if not any(m.get("role") == "tool" for m in payload["messages"]):
                    name = "get_file_context" if code else "issue_read"
                    arguments = {"path": "app.py"} if code else {"issue_number": 1}
                    deltas = [
                        ({"content": "Private tool planning"}, None),
                        (
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": name,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ]
                            },
                            None,
                        ),
                        ({}, "tool_calls"),
                    ]
                    wire = Wire(wire_chunks(deltas), None)
                elif scope == "MIXED":
                    result = json.dumps(
                        {
                            "answer": "Private investigation [1]."
                            if code
                            else f"Private investigation {url}",
                            "source_ids": ["1" if code else url],
                            "missing_information": [],
                            "context_notes": [
                                "The other specialist owns its portion of the question."
                            ],
                        }
                    )
                    wire = Wire(
                        wire_chunks([({"content": result}, None), ({}, "stop")]), None
                    )
                elif repair and len(requests) == 3:
                    wire = Wire(
                        wire_chunks(
                            [
                                ({"content": "Unmarked final response"}, None),
                                ({}, "stop"),
                            ]
                        ),
                        None,
                    )
                else:
                    wire = Wire(
                        wire_chunks(
                            [
                                ({"content": "<fi"}, None),
                                ({"content": "nal_answer>"}, None),
                                ({"content": "Live issue "}, None),
                                ({"content": url}, None),
                                ({}, "stop"),
                            ]
                        ),
                        pause,
                    )
            elif scope == "GITHUB":
                wire = Wire(
                    wire_chunks(
                        [
                            ({"content": "<fi"}, None),
                            ({"content": "nal_answer>"}, None),
                            ({"content": "Live issue "}, None),
                            ({"content": url}, None),
                            ({}, "stop"),
                        ]
                    ),
                    pause,
                )
            else:
                wire = Wire(
                    wire_chunks(
                        [
                            ({"content": "<fi"}, None),
                            ({"content": "nal_answer>"}, None),
                            ({"content": "Code [1] and "}, None),
                            ({"content": url}, None),
                            ({"content": "</final_"}, None),
                            ({"content": "answer>\n"}, None),
                            ({}, "stop"),
                        ]
                    ),
                    pause,
                )
            wires.append(wire)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=wire
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            native = chat.ChatOpenAI
            monkeypatch.setattr(
                chat,
                "ChatOpenAI",
                lambda **kwargs: native(http_async_client=client, **kwargs),
            )
            events = []
            async for event in chat.stream_repository_chat(
                ChatPool(database), "repo", "conversation", "user", "message"
            ):
                events.append(event)
                if event.startswith("event: delta") and not pause.is_set():
                    assert not wires[-1].finished
                    assert len(lifecycle) == (1 if scope == "GITHUB" else 2)
                    assert database.active == 0
                    pause.set()
            text = "".join(
                json.loads(e.split("data: ")[1])["text"]
                for e in events
                if e.startswith("event: delta")
            )
            assert (
                "Private" not in text
                and "final_answer" not in text
                and '"source_ids"' not in text
            )
            assert text == (
                f"Live issue {url}" if scope == "GITHUB" else f"Code [1] and {url}"
            )
            assert len(requests) == ((4 if repair else 3) if scope == "GITHUB" else 6)
            assert next(iter(database.messages.values())) == {
                "status": "SUCCEEDED",
                "content": text,
            }
            assert events[-1].startswith("event: done")
            assert any('"name": "issue_read"' in e for e in events)
            assert bool(database.citations) == (scope == "MIXED")
            assert len(lifecycle) == 2

    try:
        asyncio.run(asyncio.wait_for(run(), timeout=10))
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "case", ["clarification", "router_failure", "discovery_failure"]
)
def test_clarification_and_required_failures_preserve_status_without_code_reads(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from test_tool_registry import connect_offline, offline_server, provider

    from app import code_tools

    monkeypatch.setenv("HIVE_API_KEY", "offline")
    get_settings.cache_clear()
    sessions = connect_offline(monkeypatch, {"github": offline_server([], [])})
    monkeypatch.setattr(chat, "create_github_provider", lambda *args: provider())

    def unexpected() -> Any:
        pytest.fail("Clarification/GitHub discovery must not construct embeddings.")

    monkeypatch.setattr(code_tools, "create_embedding_model", unexpected)
    database, requests = ChatDatabase(), []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = (
            "malformed private JSON"
            if case == "router_failure"
            else json.dumps(
                {
                    "scope": "GITHUB" if case == "discovery_failure" else None,
                    "clarification": None
                    if case == "discovery_failure"
                    else "Which issue?",
                }
            )
        )
        return httpx.Response(
            200,
            json={
                "id": "router",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            native = chat.ChatOpenAI
            monkeypatch.setattr(
                chat,
                "ChatOpenAI",
                lambda **kwargs: native(http_async_client=client, **kwargs),
            )
            events = [
                e
                async for e in chat.stream_repository_chat(
                    ChatPool(database), "repo", "conversation", "user", "message"
                )
            ]
            persisted = next(iter(database.messages.values()))
            assert not database.queries and not database.citations
            assert len(requests) == (2 if case == "router_failure" else 1)
            assert len(sessions) == (2 if case == "discovery_failure" else 0)
            if case == "clarification":
                assert persisted == {"status": "SUCCEEDED", "content": "Which issue?"}
                assert events[-1].startswith("event: done")
            else:
                assert persisted == {"status": "FAILED", "content": ""}
                assert events[-1].startswith("event: error")
            assert all("malformed private" not in e for e in events)

    try:
        asyncio.run(asyncio.wait_for(run(), timeout=10))
    finally:
        get_settings.cache_clear()
