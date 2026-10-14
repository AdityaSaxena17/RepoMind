"""Request-owned tool discovery, capability policy, and guarded async execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Capability = Literal["code.read", "github.read"]
DISCOVERY_TIMEOUT_SECONDS = 20
INVOCATION_TIMEOUT_SECONDS = 30
_IDENTITY_ARGUMENTS = frozenset({"user_id", "repository_id", "owner", "repo"})


class ToolPolicy(BaseModel):
    """Explicit repository-bound permission declaration; hints cannot grant it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capability: Capability
    repository_scoped: bool

    @model_validator(mode="after")
    def require_repository_scope(self) -> ToolPolicy:
        if not self.repository_scoped:
            raise ValueError("Tools must be bound to the request repository.")
        return self


def parse_tool_policy(
    meta: object, annotations: Mapping[str, object] | None = None
) -> ToolPolicy | None:
    """Validate custom MCP metadata, excluding absent or contradictory policies."""
    if not isinstance(meta, dict) or not isinstance(meta.get("repomind"), dict):
        return None
    try:
        policy = ToolPolicy.model_validate(meta["repomind"])
    except ValidationError:
        return None
    if annotations and (
        annotations.get("readOnlyHint") is False
        or annotations.get("destructiveHint") is True
    ):
        return None
    return policy


class RequestScope(BaseModel):
    """Server-authorized identity and permissions, never supplied by the model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    user_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    capabilities: frozenset[Capability]


@dataclass(frozen=True)
class MCPProvider:
    """Trusted backend connection and maximum capabilities for one provider."""

    name: str
    connection: Connection
    capabilities: frozenset[Capability]


@dataclass(frozen=True)
class ToolDescriptor:
    """Policy and native tool identity from an eligible catalog entry."""

    name: str
    description: str
    provider: str
    policy: ToolPolicy


@dataclass
class _ToolScope:
    request: RequestScope
    capabilities: frozenset[Capability]
    active: bool = True


class _GuardedTool(BaseTool):
    delegate: BaseTool = Field(exclude=True, repr=False)
    descriptor: ToolDescriptor = Field(exclude=True)
    scope: _ToolScope = Field(exclude=True, repr=False)
    local_lock: asyncio.Lock = Field(exclude=True, repr=False)
    authorize: Callable[[RequestScope], Awaitable[None]] | None = Field(
        default=None, exclude=True, repr=False
    )

    @property
    def tool_call_schema(self) -> Any:
        return self.delegate.tool_call_schema

    @property
    def _injected_args_keys(self) -> frozenset[str]:
        return self.delegate._injected_args_keys

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise ToolException("Registry tools require async invocation.")

    async def ainvoke(
        self,
        input: str | dict[str, Any] | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        call_id = input.get("id") if isinstance(input, dict) else None
        if not isinstance(input, dict) or input.get("type") != "tool_call":
            call_id = None
        return await self._execute(
            lambda: self.delegate.ainvoke(input, config, **kwargs),
            call_id,
        )

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute(
            lambda: self.delegate.arun(*args, **kwargs), kwargs.get("tool_call_id")
        )

    async def _execute(
        self, invoke: Callable[[], Awaitable[Any]], call_id: str | None
    ) -> Any:
        try:
            with anyio.fail_after(INVOCATION_TIMEOUT_SECONDS):
                async with AsyncExitStack() as stack:
                    if self.authorize is not None:
                        await stack.enter_async_context(self.local_lock)
                    if not self.scope.active:
                        raise ToolException("Tool scope is closed.")
                    capability = self.descriptor.policy.capability
                    if (
                        capability not in self.scope.capabilities
                        or capability not in self.scope.request.capabilities
                    ):
                        raise PermissionError("Tool capability is not authorized.")
                    if self.authorize is not None:
                        await self.authorize(self.scope.request)
                        if not self.scope.active:
                            raise ToolException(
                                "Tool scope closed during authorization."
                            )
                    # Delegate the complete native invocation, including ToolCall and
                    # config, so LangChain retains artifacts and callback semantics.
                    return await invoke()
        except Exception as error:
            context = f"Tool '{self.name}' from '{self.descriptor.provider}'"
            if isinstance(error, TimeoutError):
                message = f"{context} timed out."
            elif isinstance(error, PermissionError):
                message = f"{context} is not authorized."
            elif not self.scope.active:
                message = f"{context} cannot run after its scope closes."
            else:
                message = f"{context} failed."
            if call_id is not None:
                return ToolMessage(
                    content=message,
                    tool_call_id=call_id,
                    name=self.name,
                    status="error",
                )
            raise ToolException(message) from error


class ToolRegistry:
    """Own fresh catalogs and sessions for one immutable authorized request."""

    def __init__(
        self, request: RequestScope, providers: tuple[MCPProvider, ...] = ()
    ) -> None:
        names = [provider.name for provider in providers]
        if any(not name or name == "local" for name in names) or len(names) != len(
            set(names)
        ):
            raise ValueError("Provider names must be unique and nonempty.")
        self.request = request
        self.providers = providers
        self._local: list[
            tuple[BaseTool, ToolDescriptor, Callable[[RequestScope], Awaitable[None]]]
        ] = []
        self._names: dict[str, str] = {}
        self._local_lock = asyncio.Lock()

    def register_local(
        self,
        tool: BaseTool,
        policy: ToolPolicy,
        authorize: Callable[[RequestScope], Awaitable[None]],
    ) -> None:
        """Register a request-bound tool; authorize must raise when access is denied."""
        if not callable(authorize):
            raise ValueError("Local tools require an authorization callback.")
        if tool.name in self._names:
            raise ValueError(f"Tool name collision: {tool.name}.")
        if _IDENTITY_ARGUMENTS.intersection(tool.args):
            raise ValueError("Tool identity must be bound outside model arguments.")
        descriptor = ToolDescriptor(tool.name, tool.description, "local", policy)
        self._names[tool.name] = "local"
        self._local.append((tool, descriptor, authorize))

    @asynccontextmanager
    async def scoped_tools(
        self, capabilities: frozenset[Capability]
    ) -> AsyncIterator[list[BaseTool]]:
        """Discover selected providers and keep guarded tools alive until scope exit."""
        if not capabilities <= self.request.capabilities:
            raise PermissionError("Requested tool capabilities are not authorized.")
        state = _ToolScope(self.request, capabilities)
        discovery_complete = False
        try:
            # Keep this cancel scope outside all MCP contexts: the SDK enters
            # task groups which must exit in LIFO order in this same owning task.
            with anyio.fail_after(DISCOVERY_TIMEOUT_SECONDS) as deadline:
                stack = AsyncExitStack()
                try:
                    selected: list[
                        tuple[
                            BaseTool,
                            ToolDescriptor,
                            Callable[[RequestScope], Awaitable[None]] | None,
                        ]
                    ] = [
                        registration
                        for registration in self._local
                        if registration[1].policy.capability in capabilities
                    ]
                    for provider in self.providers:
                        allowed = provider.capabilities & capabilities
                        if not allowed:
                            continue
                        client = MultiServerMCPClient(
                            {provider.name: provider.connection},
                            handle_tool_errors=True,
                        )
                        session = await stack.enter_async_context(
                            client.session(provider.name)
                        )
                        catalog = await load_mcp_tools(
                            session, server_name=provider.name, handle_tool_errors=True
                        )
                        eligible = []
                        for tool in catalog:
                            metadata = tool.metadata or {}
                            policy = parse_tool_policy(metadata.get("_meta"), metadata)
                            if policy is None or policy.capability not in allowed:
                                continue
                            eligible.append(
                                (
                                    tool,
                                    ToolDescriptor(
                                        tool.name,
                                        tool.description,
                                        provider.name,
                                        policy,
                                    ),
                                    None,
                                )
                            )
                        if not eligible:
                            raise ValueError(
                                f"Provider '{provider.name}' has no eligible tools."
                            )
                        batch_names: set[str] = set()
                        for tool, _, _ in eligible:
                            owner = self._names.get(tool.name)
                            if tool.name in batch_names or owner not in (
                                None,
                                provider.name,
                            ):
                                raise ValueError(f"Tool name collision: {tool.name}.")
                            batch_names.add(tool.name)
                        self._names.update(
                            {name: provider.name for name in batch_names}
                        )
                        selected.extend(eligible)
                    tools: list[BaseTool] = [
                        _GuardedTool(
                            name=tool.name,
                            description=tool.description,
                            args_schema=tool.args_schema,
                            metadata=tool.metadata,
                            tags=tool.tags,
                            return_direct=tool.return_direct,
                            response_format=tool.response_format,
                            extras=tool.extras,
                            delegate=tool,
                            descriptor=descriptor,
                            scope=state,
                            local_lock=self._local_lock,
                            authorize=authorize,
                        )
                        for tool, descriptor, authorize in selected
                    ]
                    discovery_complete = True
                    deadline.deadline = float("inf")
                    yield tools
                finally:
                    state.active = False
                    await stack.aclose()
        except Exception as error:
            if discovery_complete:
                raise
            raise ToolException("Required tool discovery failed.") from error
