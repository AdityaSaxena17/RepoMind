"""Production chat streaming, request preparation and transactional persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from pathlib import Path
from typing import Any, Literal, cast

import anyio
import asyncpg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent
from langchain_mcp_adapters.sessions import Connection
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from redis.asyncio import Redis

from .agents import AgentLimits
from .code_tools import CodeTools
from .config import get_settings
from .observability import RequestLoggingCallback, log_event, request_id_context
from .orchestration import (
    ROUTER_TIMEOUT_SECONDS,
    WorkflowResult,
    create_repository_graph,
)
from .tool_errors import AuthorizationError
from .tool_registry import (
    Capability,
    MCPProvider,
    RequestScope,
    ToolRegistry,
    parse_tool_policy,
)

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15.0


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one server-sent event with JSON data."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def github_line_url(
    html_url: str | None, ref: str, path: str, start_line: int, end_line: int
) -> str | None:
    """Format code citation URLs using the historical benchmark convention."""
    if not html_url:
        return None
    return f"{html_url}/blob/{ref}/{path}#L{start_line}-L{end_line}"


def create_chat_model(
    *, purpose: Literal["answer", "router", "synthesis"] = "answer"
) -> ChatOpenAI:
    """Configure Hive reasoning by role; preserve explicit OpenAI configuration."""
    settings = get_settings()
    routing = purpose == "router"

    if settings.hive_api_key:
        return ChatOpenAI(
            model=settings.hive_chat_model,
            api_key=SecretStr(settings.hive_api_key),
            base_url="https://api.thehive.ai/api/v3",
            streaming=not routing,
            disable_streaming=routing,
            # Hive includes usage itself and serves Chat Completions, not Responses.
            stream_usage=False,
            use_responses_api=False,
            reasoning_effort=(
                "none"
                if routing
                else settings.synthesis_reasoning_effort
                if purpose == "synthesis"
                else "high"
            ),
            max_completion_tokens=512 if routing else 16384,
            extra_body=None if routing else {"max_thinking_tokens": 8192},
            timeout=ROUTER_TIMEOUT_SECONDS if routing else 120,
            max_retries=0 if routing else 1,
            temperature=0 if routing else None,
        )

    if not settings.openai_api_key:
        raise RuntimeError(
            "HIVE_API_KEY or OPENAI_API_KEY is required for repository chat."
        )

    if not settings.openai_chat_model:
        raise RuntimeError("OPENAI_CHAT_MODEL is required for repository chat.")

    if routing:
        return ChatOpenAI(
            model=settings.openai_chat_model,
            streaming=False,
            disable_streaming=True,
            stream_usage=False,
            use_responses_api=False,
            max_completion_tokens=512,
            timeout=ROUTER_TIMEOUT_SECONDS,
            max_retries=0,
            temperature=0,
        )

    return ChatOpenAI(
        model=settings.openai_chat_model,
        streaming=True,
        stream_usage=True,
        use_responses_api=True,
    )


def create_github_provider(
    user_id: str, repository_id: str, request_id: str | None = None
) -> MCPProvider:
    """Configure our MCP provider; the registry opens its request-owned sessions."""
    env = dict(os.environ)
    env["REPOMIND_MCP_USER_ID"] = user_id
    env["REPOMIND_MCP_REPOSITORY_ID"] = repository_id
    if request_id is not None:
        env["REPOMIND_MCP_REQUEST_ID"] = request_id
    connection: Connection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "app.github_mcp_server"],
        "cwd": str(Path(__file__).resolve().parents[1]),
        "env": env,
    }
    return MCPProvider("github", connection, frozenset({"github.read"}))


async def load_history(
    connection: asyncpg.Connection,
    conversation_id: str,
    current_message_id: str,
) -> list[BaseMessage]:
    """Load the latest 20 successful messages in chronological order."""
    rows = await connection.fetch(
        """
        SELECT "id", "role", "content"
        FROM "ChatMessage"
        WHERE "conversationId" = $1
          AND "id" <> $2
          AND "status" = 'SUCCEEDED'
        ORDER BY "createdAt" DESC, "id" DESC
        LIMIT 20
        """,
        conversation_id,
        current_message_id,
    )
    messages: list[BaseMessage] = []

    for row in reversed(rows):
        content = str(row["content"])
        if row["role"] == "USER":
            messages.append(HumanMessage(content=content))
        elif row["role"] == "ASSISTANT":
            messages.append(AIMessage(content=content))

    return messages


async def _fail_message(
    pool: asyncpg.Pool, message_id: str, content: str, error: str
) -> None:
    # Failure cleanup must survive disconnect cancellation, but cannot block shutdown.
    with anyio.CancelScope(shield=True):
        try:
            with anyio.fail_after(10):
                async with pool.acquire(timeout=10) as connection:
                    await connection.execute(
                        """
                        UPDATE "ChatMessage"
                        SET "content"=$2, "status"='FAILED', "errorMessage"=$3,
                            "updatedAt"=now()
                        WHERE "id"=$1 AND "status"='STREAMING'
                        """,
                        message_id,
                        content,
                        error,
                    )
        except Exception as failure:
            logger.error("Failed to persist chat failure (%s).", type(failure).__name__)


async def stream_repository_chat(
    pool: asyncpg.Pool,
    repository_id: str,
    conversation_id: str,
    user_id: str,
    message_id: str,
    *,
    redis: Redis | None = None,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream only final answer tokens, then validate and persist the completed turn."""
    assistant_id = uuid.uuid4().hex
    trace_id = request_id or assistant_id
    log_event("request", "chat", "started", request_id=trace_id)
    parts: list[str] = []
    created = committed = False
    producer: asyncio.Task[None] | None = None
    try:
        log_event("database", "chat_prepare", "started", request_id=trace_id)
        async with pool.acquire(timeout=10) as connection:
            user_message = await connection.fetchrow(
                """
                SELECT m."content" FROM "ChatMessage" m
                JOIN "ChatConversation" c ON c."id"=m."conversationId"
                WHERE m."id"=$1 AND m."conversationId"=$2
                  AND c."userId"=$3 AND c."repositoryId"=$4 AND m."role"='USER'
                """,
                message_id,
                conversation_id,
                user_id,
                repository_id,
            )
            if not user_message:
                raise AuthorizationError(
                    "Chat message was not found or is inaccessible."
                )
            repository = await connection.fetchrow(
                """
                SELECT r."indexedCommitSha", r."lastIndexedAt", r."indexStatus",
                       r."indexedChunkCount"
                FROM "Repository" r
                JOIN "UserRepository" ur ON ur."repositoryId"=r."id"
                WHERE r."id"=$1 AND ur."userId"=$2
                  AND r."accessStatus"='ACTIVE' AND ur."accessStatus"='ACTIVE'
                """,
                repository_id,
                user_id,
            )
            if not repository:
                raise AuthorizationError("Repository access is no longer active.")
            if (
                repository["indexStatus"] != "READY"
                or repository["indexedChunkCount"] <= 0
            ):
                raise ValueError(
                    "Repository must finish indexing before chat is available."
                )
            history = await load_history(connection, conversation_id, message_id)
            await connection.execute(
                """
                INSERT INTO "ChatMessage"
                  ("id","conversationId","role","content","status","updatedAt")
                VALUES ($1,$2,'ASSISTANT','','STREAMING',now())
                """,
                assistant_id,
                conversation_id,
            )
            created = True
        log_event("database", "chat_prepare", "completed", request_id=trace_id)
        yield sse_event(
            "assistant_message", {"message_id": assistant_id, "request_id": trace_id}
        )
        capabilities: frozenset[Capability] = frozenset({"code.read", "github.read"})
        registry = ToolRegistry(
            RequestScope(
                user_id=user_id, repository_id=repository_id, capabilities=capabilities
            ),
            (create_github_provider(user_id, repository_id, trace_id),),
        )
        code_tools = CodeTools(
            registry,
            None,
            str(repository["indexedCommitSha"]),
            pool=pool,
            redis=redis,
            last_indexed_at=repository["lastIndexedAt"],
            # Interactive reranking is paused for latency; offline comparisons
            # can still explicitly enable it when constructing their code tools.
            rerank=False,
        )
        settings = get_settings()
        model = create_chat_model()
        graph = create_repository_graph(
            create_chat_model(purpose="router"),
            model,
            model,
            create_chat_model(purpose="synthesis"),
            registry=registry,
            code_tools=code_tools,
            limits=AgentLimits(
                tool_calls=settings.agent_tool_call_limit,
                model_calls=settings.agent_model_call_limit,
                timeout_seconds=settings.agent_timeout_seconds,
            ),
        )
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)

        async def produce() -> None:
            # This ordinary coroutine owns the token; an ASGI generator may be
            # closed by a different task and must not own a ContextVar token.
            trace_token = request_id_context.set(trace_id)
            root_id: str | None = None
            emitted_tools: set[str] = set()
            try:
                # v2 returns a generator with cleanup, typed only as AsyncIterator.
                async with aclosing(
                    cast(
                        AsyncGenerator[StreamEvent, None],
                        graph.astream_events(
                            {
                                "question": str(user_message["content"]),
                                "history": history,
                            },
                            config={
                                "callbacks": [RequestLoggingCallback(trace_id)],
                                "metadata": {"request_id": trace_id},
                            },
                            version="v2",
                            include_types=["chain", "tool"],
                            include_names=[
                                "repomind_answer_delta",
                                "repomind_progress",
                            ],
                        ),
                    )
                ) as stream:
                    async for event in stream:
                        kind = event["event"]
                        if kind == "on_chain_start" and not event["parent_ids"]:
                            root_id = event["run_id"]
                        elif kind == "on_tool_start":
                            metadata = event.get("metadata", {})
                            policy = parse_tool_policy(metadata.get("_meta"), metadata)
                            if (
                                policy is not None
                                and policy.capability in capabilities
                                and event["run_id"] not in emitted_tools
                            ):
                                emitted_tools.add(event["run_id"])
                                await queue.put(("tool", event["name"]))
                        elif (
                            kind == "on_custom_event"
                            and event["name"] == "repomind_progress"
                        ):
                            await queue.put(("progress", event["data"]))
                        elif (
                            kind == "on_custom_event"
                            and event["name"] == "repomind_answer_delta"
                        ):
                            data = event["data"]
                            text = data.get("text")
                            if data.get("role") in {
                                "CODE",
                                "GITHUB",
                                "SYNTHESIS",
                            } and isinstance(text, str):
                                await queue.put(
                                    (
                                        "delta",
                                        {
                                            "text": text,
                                            "native": data.get("native") is True,
                                        },
                                    )
                                )
                        elif kind == "on_chain_end" and event["run_id"] == root_id:
                            await queue.put(
                                (
                                    "result",
                                    WorkflowResult.model_validate(
                                        event["data"]["output"]["result"]
                                    ),
                                )
                            )
            except Exception as failure:
                await queue.put(("failure", failure))
            finally:
                request_id_context.reset(trace_token)

        producer = asyncio.create_task(produce())
        result: WorkflowResult | None = None
        while result is None:
            try:
                kind, value = await asyncio.wait_for(
                    queue.get(), timeout=HEARTBEAT_SECONDS
                )
            except asyncio.TimeoutError:
                if producer.done():
                    await producer
                    raise RuntimeError(
                        "Workflow ended without a final result."
                    ) from None
                yield ": heartbeat\n\n"
                continue
            if kind == "failure":
                raise value
            if kind == "result":
                result = value
            else:
                if kind == "delta":
                    parts.append(value["text"])
                    yield sse_event("delta", value)
                elif kind == "progress":
                    yield sse_event("progress", value)
                else:
                    yield sse_event(kind, {"name": value})
        await producer
        if result.status == "failed":
            failure = result.errors[-1] if result.errors else None
            raise RuntimeError(
                failure.message if failure else "Investigation could not be completed."
            )
        if result.status == "clarification":
            parts.append(result.answer)
            yield sse_event("delta", {"text": result.answer})
        if "".join(parts) != result.answer:
            raise RuntimeError(
                "Streamed answer does not match the validated workflow result."
            )
        citations = [
            {
                "id": uuid.uuid4().hex,
                "source_index": source.source_index,
                "path": source.path,
                "start_line": source.start_line,
                "end_line": source.end_line,
                "url": source.url,
            }
            for source in result.evidence
            if source.id in result.source_ids
        ]
        log_event("database", "chat_persist", "started", request_id=trace_id)
        async with (
            pool.acquire(timeout=10) as connection,
            connection.transaction(),
        ):
            allowed = await connection.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM "Repository" r
                    JOIN "UserRepository" ur ON ur."repositoryId"=r."id"
                    JOIN "ChatConversation" c ON c."repositoryId"=r."id"
                    WHERE r."id"=$1 AND ur."userId"=$2 AND c."id"=$3
                      AND c."userId"=$2 AND r."accessStatus"='ACTIVE'
                      AND ur."accessStatus"='ACTIVE'
                )
                """,
                repository_id,
                user_id,
                conversation_id,
            )
            if not allowed:
                raise AuthorizationError(
                    "Repository access was revoked during the answer."
                )
            for citation in citations:
                await connection.execute(
                    """
                    INSERT INTO "ChatMessageCitation"
                      ("id","messageId","sourceIndex","path","startLine","endLine","url")
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    """,
                    citation["id"],
                    assistant_id,
                    citation["source_index"],
                    citation["path"],
                    citation["start_line"],
                    citation["end_line"],
                    citation["url"],
                )
            await connection.execute(
                """
                UPDATE "ChatMessage" SET "content"=$2, "status"='SUCCEEDED',
                    "errorMessage"=NULL, "updatedAt"=now() WHERE "id"=$1
                """,
                assistant_id,
                result.answer,
            )
            await connection.execute(
                'UPDATE "ChatConversation" SET "updatedAt"=now() WHERE "id"=$1',
                conversation_id,
            )
        committed = True
        log_event("database", "chat_persist", "completed", request_id=trace_id)
        log_event(
            "request",
            "chat",
            "completed",
            request_id=trace_id,
            route=result.decision.scope,
            outcome=result.status,
        )
        logger.info(
            "Chat finished: scope=%s status=%s model_calls=%d cache=%s",
            result.decision.scope,
            result.status,
            result.model_calls,
            code_tools.cache_stats,
        )
        yield sse_event("citations", {"citations": citations})
        yield sse_event("done", {"message_id": assistant_id})
    except (asyncio.CancelledError, GeneratorExit):
        log_event("request", "chat", "cancelled", request_id=trace_id)
        if created and not committed:
            await _fail_message(
                pool, assistant_id, "".join(parts), "Answer canceled before completion."
            )
        raise
    except Exception as failure:
        log_event(
            "request",
            "chat",
            "failed",
            request_id=trace_id,
            error_type=type(failure).__name__,
        )
        if producer is not None and not producer.done():
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
        message = (
            str(failure)
            if isinstance(failure, (AuthorizationError, RuntimeError, ValueError))
            and failure.__class__.__module__.startswith(("app", "builtins"))
            else f"Repository chat failed ({type(failure).__name__})."
        )
        message = message.replace("\n", " ")[:300]
        if created and not committed:
            await _fail_message(pool, assistant_id, "".join(parts), message)
        yield sse_event("error", {"message": message})
    finally:
        if producer is not None and not producer.done():
            producer.cancel()
            with anyio.CancelScope(shield=True):
                with suppress(asyncio.CancelledError):
                    await producer
