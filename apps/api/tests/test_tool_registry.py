import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from mcp import ClientSession, types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from app import chat, tool_registry
from app.tool_registry import (
    MCPProvider,
    RequestScope,
    ToolPolicy,
    ToolRegistry,
    parse_tool_policy,
)

CODE = frozenset({"code.read"})
GITHUB = frozenset({"github.read"})
META = {"repomind": {"capability": "github.read", "repository_scoped": True}}
ANNOTATIONS = types.ToolAnnotations(readOnlyHint=True, destructiveHint=False)


def request(repository: str = "repo") -> RequestScope:
    return RequestScope(
        user_id="user", repository_id=repository, capabilities=CODE | GITHUB
    )


def provider(name: str = "github") -> MCPProvider:
    return MCPProvider(
        name, {"transport": "stdio", "command": "unused-offline"}, GITHUB
    )


async def authorize(_scope: RequestScope) -> None:
    return None


def local_tool(name: str = "search_code", coroutine: Any = None) -> StructuredTool:
    async def execute(query: str = "default") -> str:
        return query

    return StructuredTool.from_function(
        coroutine=coroutine or execute, name=name, description="Search code."
    )


def offline_server(
    catalog: list[types.Tool], calls: list[tuple[str, dict[str, Any]]]
) -> Server:
    server = Server("offline-registry")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return catalog

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        calls.append((name, arguments))
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text="https://example.test/issue/1")
            ],
            structuredContent={"html_url": "https://example.test/issue/1"},
            isError=arguments.get("fail", False),
        )

    return server


def definition(name: str, meta: Any = META, **kwargs: Any) -> types.Tool:
    return types.Tool(
        name=name,
        description="Offline repository tool.",
        inputSchema={"type": "object", "properties": {}},
        _meta=meta,
        annotations=kwargs.pop("annotations", ANNOTATIONS),
        **kwargs,
    )


def connect_offline(
    monkeypatch: pytest.MonkeyPatch, servers: dict[str, Server]
) -> list[tuple[str, str, object]]:
    events: list[tuple[str, str, object]] = []

    class Client:
        def __init__(self, connections: object, **_kwargs: object) -> None:
            self.connections = connections

        @asynccontextmanager
        async def session(self, name: str) -> AsyncIterator[ClientSession]:
            task = asyncio.current_task()
            events.append(("open", name, task))
            try:
                async with create_connected_server_and_client_session(
                    servers[name]
                ) as session:
                    yield session
            finally:
                assert asyncio.current_task() is task
                events.append(("close", name, task))

    monkeypatch.setattr(tool_registry, "MultiServerMCPClient", Client)
    return events


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {"capability": "github.read", "repository_scoped": True},
        {"repomind": None},
        {"repomind": {"capability": "github.write", "repository_scoped": True}},
        {"repomind": {"capability": "github.read"}},
        {"repomind": {"capability": "github.read", "repository_scoped": False}},
        {"repomind": {"capability": "github.read", "repository_scoped": 1}},
        {"repomind": {"capability": "github.read", "repository_scoped": "true"}},
        {"repomind": {**META["repomind"], "provider": "trusted"}},
    ],
)
def test_invalid_or_forged_policy_is_excluded(meta: object) -> None:
    assert parse_tool_policy(meta, {"readOnlyHint": True}) is None


def test_policy_and_request_are_strict_and_immutable() -> None:
    assert parse_tool_policy(META).capability == "github.read"
    assert parse_tool_policy(META, {"readOnlyHint": False}) is None
    assert parse_tool_policy(META, {"destructiveHint": True}) is None
    scope = request()
    with pytest.raises(ValidationError):
        scope.repository_id = "other"
    with pytest.raises(ValidationError):
        RequestScope(user_id="user", repository_id="repo", capabilities={"code.read"})


def test_native_mcp_round_trip_pagination_and_new_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    server = offline_server([], calls)
    cursors: list[str | None] = []

    @server.list_tools()
    async def list_tools(req: types.ListToolsRequest) -> types.ListToolsResult:
        cursor = req.params.cursor if req.params else None
        cursors.append(cursor)
        if cursor is None:
            return types.ListToolsResult(
                tools=[definition("new_github_reader")], nextCursor="page-two"
            )
        return types.ListToolsResult(tools=[definition("another_reader")])

    events = connect_offline(monkeypatch, {"github": server})
    callback_events: list[tuple[str, object]] = []

    class Callback(AsyncCallbackHandler):
        async def on_tool_start(self, *_args: Any, **kwargs: Any) -> None:
            callback_events.append(("start", kwargs["tags"]))

        async def on_tool_end(self, output: Any, **_kwargs: Any) -> None:
            callback_events.append(("end", output))

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        async with registry.scoped_tools(GITHUB) as tools:
            assert [tool.name for tool in tools] == [
                "new_github_reader",
                "another_reader",
            ]
            result = await tools[0].ainvoke(
                {
                    "name": tools[0].name,
                    "args": {},
                    "id": "native-id",
                    "type": "tool_call",
                },
                config={"callbacks": [Callback()], "tags": ["parent-callback"]},
            )
            assert isinstance(result, ToolMessage)
            assert result.tool_call_id == "native-id"
            assert result.status == "success"
            assert result.artifact["structured_content"] == {
                "html_url": "https://example.test/issue/1"
            }
            assert "https://example.test/issue/1" in str(result.content)
            failed = await tools[1].ainvoke(
                {
                    "name": tools[1].name,
                    "args": {"fail": True},
                    "id": "failed-id",
                    "type": "tool_call",
                }
            )
            assert failed.status == "error"
            assert failed.tool_call_id == "failed-id"
            assert "https://example.test/issue/1" in str(failed.content)
        with pytest.raises(ToolException, match="scope closes"):
            await tools[0].ainvoke({})

    asyncio.run(run())
    assert cursors == [None, "page-two"]
    assert [name for name, _ in calls] == ["new_github_reader", "another_reader"]
    assert [event[:2] for event in events] == [("open", "github"), ("close", "github")]
    assert callback_events[0] == ("start", ["parent-callback"])
    assert callback_events[1][0] == "end"


def test_metadata_normalization_and_provider_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = [
        definition("eligible", {**META, "provider": "local"}),
        definition("missing", None),
        definition("annotation_only", {}),
        definition("root_forgery", META["repomind"]),
        definition(
            "write", {"repomind": {**META["repomind"], "capability": "github.write"}}
        ),
        definition(
            "wrong_provider",
            {"repomind": {"capability": "code.read", "repository_scoped": True}},
        ),
        definition(
            "contradictory", annotations=types.ToolAnnotations(readOnlyHint=False)
        ),
    ]
    connect_offline(monkeypatch, {"github": offline_server(catalog, [])})

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        async with registry.scoped_tools(CODE | GITHUB) as tools:
            assert [tool.name for tool in tools] == ["eligible"]
            assert tools[0].descriptor.provider == "github"

    asyncio.run(run())


def test_local_authorization_revocation_lifetime_and_code_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_provider(*_args: object, **_kwargs: object) -> None:
        pytest.fail("CODE scopes must not open GitHub MCP.")

    monkeypatch.setattr(tool_registry, "MultiServerMCPClient", unexpected_provider)

    async def run() -> None:
        allowed = True
        executions: list[str] = []
        checks: list[RequestScope] = []

        async def check(scope: RequestScope) -> None:
            checks.append(scope)
            if not allowed:
                raise PermissionError("private authorization detail")

        async def execute(query: str = "default") -> str:
            executions.append(query)
            return query

        scope = request()
        registry = ToolRegistry(scope, (provider(),))
        registry.register_local(
            local_tool(coroutine=execute),
            ToolPolicy(capability="code.read", repository_scoped=True),
            check,
        )
        async with registry.scoped_tools(CODE) as tools:
            assert await tools[0].ainvoke({}) == "default"
            allowed = False
            denied = await tools[0].ainvoke(
                {"name": tools[0].name, "args": {}, "id": "denied", "type": "tool_call"}
            )
            assert denied.status == "error"
            assert denied.tool_call_id == "denied"
            assert "not authorized" in denied.content
            assert "private" not in denied.content
            with pytest.raises(ToolException, match="not authorized"):
                await tools[0].arun({})
        with pytest.raises(ToolException, match="scope closes"):
            await tools[0].ainvoke({})
        assert executions == ["default"]
        assert checks == [scope, scope, scope]
        denied_request = ToolRegistry(scope.model_copy(update={"capabilities": CODE}))
        with pytest.raises(PermissionError):
            async with denied_request.scoped_tools(GITHUB):
                pytest.fail("Unauthorized scope entered.")

    asyncio.run(run())


@pytest.mark.parametrize("collision", ["duplicate", "local", "providers"])
def test_name_collisions_fail_without_renaming(
    monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    server = offline_server(
        [definition("same"), definition("same")]
        if collision == "duplicate"
        else [definition("same")],
        [],
    )
    events = connect_offline(monkeypatch, {"github": server, "second": server})

    async def run() -> None:
        providers = (
            (provider(), provider("second"))
            if collision == "providers"
            else (provider(),)
        )
        registry = ToolRegistry(request(), providers)
        if collision == "local":
            registry.register_local(
                local_tool("same"),
                ToolPolicy(capability="code.read", repository_scoped=True),
                authorize,
            )
        with pytest.raises(ToolException) as error:
            async with registry.scoped_tools(GITHUB):
                pytest.fail("Colliding catalog must not be exposed.")
        assert "collision" in str(error.value.__cause__)

    asyncio.run(run())
    assert sum(event[0] == "open" for event in events) == sum(
        event[0] == "close" for event in events
    )


def test_local_registration_requires_auth_and_bound_identity() -> None:
    registry = ToolRegistry(request())
    policy = ToolPolicy(capability="code.read", repository_scoped=True)
    registry.register_local(local_tool(), policy, authorize)
    with pytest.raises(ValueError, match="collision"):
        registry.register_local(local_tool(), policy, authorize)
    with pytest.raises(ValueError, match="authorization"):
        registry.register_local(local_tool("other"), policy, None)

    async def unbound(repository_id: str) -> str:
        return repository_id

    with pytest.raises(ValueError, match="identity"):
        registry.register_local(local_tool("unbound", unbound), policy, authorize)
    with pytest.raises(ValueError, match="Provider names"):
        ToolRegistry(request(), (provider(), provider()))


@pytest.mark.parametrize(
    "failure", ["discovery", "discovery_timeout", "empty", "consumer", "cancel"]
)
def test_session_cleanup_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    server = offline_server([] if failure == "empty" else [definition("eligible")], [])
    events = connect_offline(monkeypatch, {"github": server})
    native_load = tool_registry.load_mcp_tools

    async def load(*args: Any, **kwargs: Any) -> Any:
        if failure == "discovery":
            raise ConnectionError("offline failure")
        if failure == "discovery_timeout":
            await asyncio.sleep(10)
        return await native_load(*args, **kwargs)

    monkeypatch.setattr(tool_registry, "load_mcp_tools", load)
    monkeypatch.setattr(tool_registry, "DISCOVERY_TIMEOUT_SECONDS", 0.03)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        expected = (
            asyncio.CancelledError
            if failure == "cancel"
            else (RuntimeError if failure == "consumer" else ToolException)
        )
        with pytest.raises(expected):
            async with registry.scoped_tools(GITHUB):
                if failure == "consumer":
                    raise RuntimeError("investigation failed")
                if failure == "cancel":
                    raise asyncio.CancelledError
                pytest.fail("Discovery failure entered scope.")

    asyncio.run(run())
    assert [event[:2] for event in events] == [("open", "github"), ("close", "github")]


@pytest.mark.parametrize("cancel", [False, True])
def test_native_mcp_invocation_timeout_and_task_cancellation(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    server = offline_server([definition("slow_reader")], [])
    started = asyncio.Event()

    @server.call_tool()
    async def slow(_name: str, _arguments: dict[str, Any]) -> dict[str, str]:
        started.set()
        await asyncio.sleep(10)
        return {"result": "too late"}

    events = connect_offline(monkeypatch, {"github": server})
    monkeypatch.setattr(tool_registry, "INVOCATION_TIMEOUT_SECONDS", 0.05)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        async with registry.scoped_tools(GITHUB) as tools:
            task = asyncio.create_task(
                tools[0].ainvoke(
                    {
                        "name": "slow_reader",
                        "args": {},
                        "id": "slow",
                        "type": "tool_call",
                    }
                )
            )
            await started.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.status == "error"
                assert result.tool_call_id == "slow"
                assert "timed out" in result.content

    asyncio.run(run())
    assert [event[:2] for event in events] == [("open", "github"), ("close", "github")]


@pytest.mark.parametrize("during_discovery", [False, True])
def test_cancelled_owner_closes_mcp_in_owning_task(
    monkeypatch: pytest.MonkeyPatch, during_discovery: bool
) -> None:
    server = offline_server([definition("eligible")], [])
    events = connect_offline(monkeypatch, {"github": server})
    started = asyncio.Event()
    native_load = tool_registry.load_mcp_tools

    async def load(*args: Any, **kwargs: Any) -> Any:
        if during_discovery:
            started.set()
            await asyncio.sleep(10)
        return await native_load(*args, **kwargs)

    monkeypatch.setattr(tool_registry, "load_mcp_tools", load)

    async def run() -> None:
        async def investigation() -> None:
            registry = ToolRegistry(request(), (provider(),))
            async with registry.scoped_tools(GITHUB):
                started.set()
                await asyncio.sleep(10)

        owner = asyncio.create_task(investigation())
        await started.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner

    asyncio.run(run())
    assert [event[:2] for event in events] == [("open", "github"), ("close", "github")]


@pytest.mark.parametrize(
    "failure", ["transport", "invocation_timeout", "auth_timeout", "cancel"]
)
def test_invocation_failure_deadlines_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(tool_registry, "INVOCATION_TIMEOUT_SECONDS", 0.02)
    executions: list[str] = []

    async def check(_scope: RequestScope) -> None:
        if failure == "auth_timeout":
            await asyncio.sleep(10)

    async def execute(query: str = "default") -> str:
        executions.append(query)
        if failure == "transport":
            raise ConnectionError("private transport detail")
        if failure == "cancel":
            raise asyncio.CancelledError
        await asyncio.sleep(10)
        return query

    async def run() -> None:
        registry = ToolRegistry(request())
        registry.register_local(
            local_tool(coroutine=execute),
            ToolPolicy(capability="code.read", repository_scoped=True),
            check,
        )
        async with registry.scoped_tools(CODE) as tools:
            if failure == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await tools[0].ainvoke({})
                return
            result = await tools[0].arun({}, tool_call_id="call-failed")
            assert result.status == "error"
            assert result.tool_call_id == "call-failed"
            assert "private" not in result.content
            assert ("timed out" in result.content) is (failure != "transport")
            with pytest.raises(ToolException):
                await tools[0].ainvoke({})
        if failure == "auth_timeout":
            assert executions == []

    asyncio.run(run())


def test_local_calls_can_overlap_and_requests_are_isolated() -> None:
    async def run() -> None:
        active = 0
        maximum = 0
        seen: list[str] = []

        async def check(scope: RequestScope) -> None:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            seen.append(scope.repository_id)
            await asyncio.sleep(0)

        async def execute(query: str = "default") -> str:
            nonlocal active
            await asyncio.sleep(0)
            active -= 1
            return query

        policy = ToolPolicy(capability="code.read", repository_scoped=True)
        registry = ToolRegistry(request())
        registry.register_local(local_tool(coroutine=execute), policy, check)
        async with (
            registry.scoped_tools(CODE) as first,
            registry.scoped_tools(CODE) as second,
        ):
            await asyncio.gather(first[0].ainvoke({}), second[0].ainvoke({}))
        assert maximum == 2
        assert seen == ["repo", "repo"]
        other = ToolRegistry(request("other"))
        other.register_local(local_tool(coroutine=execute), policy, check)
        async with other.scoped_tools(CODE) as tools:
            await tools[0].ainvoke({})
        assert seen[-1] == "other"

    asyncio.run(run())


def test_local_authorization_cannot_outlive_scope() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        executions: list[str] = []

        async def check(_scope: RequestScope) -> None:
            started.set()
            await release.wait()

        async def execute(query: str = "default") -> str:
            executions.append(query)
            return query

        registry = ToolRegistry(request())
        registry.register_local(
            local_tool(coroutine=execute),
            ToolPolicy(capability="code.read", repository_scoped=True),
            check,
        )
        async with registry.scoped_tools(CODE) as tools:
            invocation = asyncio.create_task(
                tools[0].arun({}, tool_call_id="authorization-pending")
            )
            await started.wait()
        release.set()
        result = await invocation
        assert result.status == "error"
        assert result.tool_call_id == "authorization-pending"
        assert "scope closes" in result.content
        assert executions == []

    asyncio.run(run())


def test_fresh_catalog_and_legacy_loader_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = [definition("initial")]
    events = connect_offline(monkeypatch, {"github": offline_server(catalog, [])})
    native_client = tool_registry.MultiServerMCPClient
    connections: list[dict[str, Any]] = []

    def client(configuration: dict[str, Any], **kwargs: Any) -> Any:
        connections.append(configuration)
        return native_client(configuration, **kwargs)

    monkeypatch.setattr(tool_registry, "MultiServerMCPClient", client)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        async with registry.scoped_tools(GITHUB) as tools:
            catalog.append(definition("new_reader"))
            assert [tool.name for tool in tools] == ["initial"]
        async with registry.scoped_tools(GITHUB) as tools:
            assert [tool.name for tool in tools] == ["initial", "new_reader"]
        async with chat.scoped_mcp_tools("user", "repo") as tools:
            assert [tool.name for tool in tools] == ["initial", "new_reader"]

    asyncio.run(run())
    assert len(events) == 6
    github = connections[-1]["github"]
    assert github["transport"] == "stdio"
    assert github["args"] == ["-m", "app.github_mcp_server"]
    assert github["env"]["REPOMIND_MCP_USER_ID"] == "user"
    assert github["env"]["REPOMIND_MCP_REPOSITORY_ID"] == "repo"


def test_required_discovery_failure_preserves_chat_status_and_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[object, ...]] = []
    events = connect_offline(monkeypatch, {"github": offline_server([], [])})

    class Connection:
        async def fetchrow(self, sql: str, *_args: object) -> dict[str, object]:
            if 'FROM "Repository"' in sql:
                return {
                    "id": "repo",
                    "htmlUrl": "https://example.test/repo",
                    "indexedCommitSha": "a" * 40,
                    "defaultBranch": "main",
                    "indexStatus": "READY",
                    "indexedChunkCount": 1,
                }
            return {"content": "question"}

        async def fetch(self, *_args: object) -> list[object]:
            return []

        async def execute(self, *args: object) -> str:
            writes.append(args)
            return "OK"

    class Retriever:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def ainvoke(self, _question: str) -> list[object]:
            return []

    def unexpected_agent(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Investigation must not run after required discovery fails.")

    monkeypatch.setattr(chat, "RepoMindParentHybridRetriever", Retriever)
    monkeypatch.setattr(chat, "create_embedding_model", lambda: object())
    monkeypatch.setattr(chat, "stream_repository_answer", unexpected_agent)

    async def run() -> list[str]:
        return [
            event
            async for event in chat.stream_repository_chat(
                Connection(), "repo", "conversation", "user", "message"
            )
        ]

    result = asyncio.run(run())
    assert [event.splitlines()[0] for event in result] == [
        "event: assistant_message",
        "event: error",
    ]
    assert "Required tool discovery failed" in result[-1]
    assert "'FAILED'" in str(writes[-1][0])
    assert [event[:2] for event in events] == [("open", "github"), ("close", "github")]
