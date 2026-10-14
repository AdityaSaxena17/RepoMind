from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import asyncpg
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from .config import get_settings
from .retrieval import RepoMindParentHybridRetriever, create_embedding_model

MCP_TOOL_NAMES = {
    "search_issues",
    "issue_read",
    "search_pull_requests",
    "pull_request_read",
}
AgentStreamMode = Literal[
    "values",
    "updates",
    "checkpoints",
    "tasks",
    "debug",
    "messages",
    "custom",
]
AGENT_STREAM_MODES: Sequence[AgentStreamMode] = (
    "messages",
    "updates",
)
AGENT_STREAM_VERSION: Literal["v2"] = "v2"

CHAT_SYSTEM_PROMPT = (
    "You are RepoMind, an AI assistant answering questions about one indexed "
    "GitHub repository. Use indexed repository context for code claims. Use the "
    "available MCP tools for live issue and pull request questions. Cite code "
    "claims with source numbers like [1] or [2]. Cite issue and pull request "
    "claims with GitHub URLs returned by tools. If the available context does "
    "not contain enough evidence, say what is missing. Never invent file paths, "
    "APIs, line numbers, issue numbers, pull request numbers, authors, labels, "
    "states, URLs, or behavior."
)


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one server-sent event with JSON data."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def redacted_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:300] if message else error.__class__.__name__


def create_chat_model() -> ChatOpenAI:
    """Create the repository chat model; OPENAI_CHAT_MODEL is intentionally required."""
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for repository chat.")

    if not settings.openai_chat_model:
        raise RuntimeError("OPENAI_CHAT_MODEL is required for repository chat.")

    return ChatOpenAI(
        model=settings.openai_chat_model,
        streaming=True,
        stream_usage=True,
        use_responses_api=True,
    )


def format_documents(documents: list[Document]) -> str:
    """Format retrieved parent windows into citation-addressable context."""
    if not documents:
        return "No indexed repository context was retrieved."

    sections: list[str] = []

    for index, document in enumerate(documents, start=1):
        path = document.metadata.get("path")
        start_line = document.metadata.get("start_line")
        end_line = document.metadata.get("end_line")
        location = (
            f"{path}:{start_line}-{end_line}"
            if isinstance(path, str)
            and isinstance(start_line, int)
            and isinstance(end_line, int)
            else "unknown"
        )
        sections.append(f"[{index}] {location}\n{document.page_content}")

    return "\n\n".join(sections)


@asynccontextmanager
async def scoped_mcp_tools(
    user_id: str,
    repository_id: str,
) -> AsyncIterator[list[Any]]:
    """Load real MCP tools from a repository-scoped stdio server session."""
    env = dict(os.environ)
    env["REPOMIND_MCP_USER_ID"] = user_id
    env["REPOMIND_MCP_REPOSITORY_ID"] = repository_id
    api_root = Path(__file__).resolve().parents[1]
    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "app.github_mcp_server"],
                "cwd": str(api_root),
                "env": env,
            },
        },
        handle_tool_errors=True,
    )

    async with client.session("github") as session:
        tools = await load_mcp_tools(session)
        yield [tool for tool in tools if tool.name in MCP_TOOL_NAMES]


def agent_messages(
    question: str,
    history: list[BaseMessage],
    documents: list[Document],
) -> list[BaseMessage]:
    """Build the message list passed to the repository chat agent."""
    return [
        *history,
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Indexed code context:\n{format_documents(documents)}"
            ),
        ),
    ]


def message_text(token: object) -> str:
    """Extract streamed assistant text from LangChain message events."""
    if isinstance(token, AIMessage | AIMessageChunk):
        return token.text

    return ""


def tool_calls(message: object) -> list[tuple[str, str]]:
    if not isinstance(message, AIMessage):
        return []

    calls: list[tuple[str, str]] = []

    for tool_call in message.tool_calls:
        name = tool_call.get("name")
        call_id = tool_call.get("id")
        if isinstance(name, str) and name in MCP_TOOL_NAMES:
            calls.append((name, str(call_id or name)))

    return calls


def github_line_url(
    html_url: str | None,
    ref: str,
    path: str,
    start_line: int,
    end_line: int,
) -> str | None:
    if not html_url:
        return None

    return f"{html_url}/blob/{ref}/{path}#L{start_line}-L{end_line}"


async def load_history(
    connection: asyncpg.Connection,
    conversation_id: str,
    current_message_id: str,
) -> list[BaseMessage]:
    """Load persisted chat history, excluding the in-flight user message."""
    rows = await connection.fetch(
        """
        SELECT "id", "role", "content"
        FROM "ChatMessage"
        WHERE "conversationId" = $1
          AND "id" <> $2
          AND "status" = 'SUCCEEDED'
        ORDER BY "createdAt" ASC
        LIMIT 20
        """,
        conversation_id,
        current_message_id,
    )
    messages: list[BaseMessage] = []

    for row in rows:
        content = str(row["content"])
        if row["role"] == "USER":
            messages.append(HumanMessage(content=content))
        elif row["role"] == "ASSISTANT":
            messages.append(AIMessage(content=content))

    return messages


async def stream_repository_answer(
    question: str,
    history: list[BaseMessage],
    documents: list[Document],
    tools: Sequence[BaseTool],
) -> AsyncIterator[tuple[Literal["delta", "tool"], str]]:
    """Run the production agent; yield text and tool activity without persistence."""
    emitted_tool_calls: set[str] = set()
    # LangGraph's compiled agent type is narrower than its documented
    # dynamic state input, so keep the untyped boundary at construction.
    agent: Any = create_agent(
        create_chat_model(),
        tools=list(tools),
        system_prompt=CHAT_SYSTEM_PROMPT,
        name="repomind",
    )
    agent_input: dict[str, Any] = {
        "messages": agent_messages(
            question,
            history,
            documents,
        ),
    }
    async for chunk in agent.astream(
        agent_input,
        stream_mode=AGENT_STREAM_MODES,
        version=AGENT_STREAM_VERSION,
    ):
        if not isinstance(chunk, dict):
            continue

        if chunk.get("type") == "messages":
            data = chunk.get("data")

            if isinstance(data, tuple) and data:
                text = message_text(data[0])

                if text:
                    yield "delta", text

        if chunk.get("type") == "updates":
            data = chunk.get("data")

            if not isinstance(data, dict):
                continue

            for update in data.values():
                if not isinstance(update, dict):
                    continue

                messages = update.get("messages")
                if not isinstance(messages, list) or not messages:
                    continue

                for name, call_id in tool_calls(messages[-1]):
                    if call_id in emitted_tool_calls:
                        continue

                    emitted_tool_calls.add(call_id)
                    yield "tool", name


async def stream_repository_chat(
    connection: asyncpg.Connection,
    repository_id: str,
    conversation_id: str,
    user_id: str,
    message_id: str,
) -> AsyncIterator[str]:
    """Stream a repository-grounded chat answer and persist the final response."""
    assistant_message_id = uuid.uuid4().hex
    answer_parts: list[str] = []
    documents: list[Document] = []

    try:
        repository = await connection.fetchrow(
            """
            SELECT r."id", r."owner", r."name", r."htmlUrl", r."indexedCommitSha",
                   r."defaultBranch", r."indexStatus", r."indexedChunkCount"
            FROM "Repository" r
            JOIN "UserRepository" ur ON ur."repositoryId" = r."id"
            WHERE r."id" = $1
              AND ur."userId" = $2
              AND r."accessStatus" = 'ACTIVE'
              AND ur."accessStatus" = 'ACTIVE'
            """,
            repository_id,
            user_id,
        )

        if not repository:
            raise RuntimeError("Repository is not active or no longer accessible.")

        if (
            repository["indexStatus"] != "READY"
            or int(repository["indexedChunkCount"]) <= 0
        ):
            raise RuntimeError(
                "Repository must finish indexing before chat is available.",
            )

        user_message = await connection.fetchrow(
            """
            SELECT m."content"
            FROM "ChatMessage" m
            JOIN "ChatConversation" c ON c."id" = m."conversationId"
            WHERE m."id" = $1
              AND m."conversationId" = $2
              AND c."userId" = $3
              AND c."repositoryId" = $4
              AND m."role" = 'USER'
            """,
            message_id,
            conversation_id,
            user_id,
            repository_id,
        )

        if not user_message:
            raise RuntimeError("Chat message was not found.")

        await connection.execute(
            """
            INSERT INTO "ChatMessage"
              ("id", "conversationId", "role", "content", "status", "updatedAt")
            VALUES ($1, $2, 'ASSISTANT', '', 'STREAMING', now())
            """,
            assistant_message_id,
            conversation_id,
        )
        yield sse_event("assistant_message", {"message_id": assistant_message_id})

        retriever = RepoMindParentHybridRetriever(
            connection=connection,
            embeddings=create_embedding_model(),
            repository_id=repository_id,
            rerank=get_settings().repository_rerank,
        )
        documents = await retriever.ainvoke(str(user_message["content"]))
        history = await load_history(connection, conversation_id, message_id)
        async with scoped_mcp_tools(user_id, repository_id) as mcp_tools:
            async for event, value in stream_repository_answer(
                str(user_message["content"]),
                history,
                documents,
                mcp_tools,
            ):
                if event == "delta":
                    answer_parts.append(value)
                yield sse_event(event, {"text" if event == "delta" else "name": value})

        final_answer = "".join(answer_parts).strip()

        if not final_answer:
            final_answer = "I could not generate an answer from the indexed context."

        await connection.execute(
            """
            UPDATE "ChatMessage"
            SET "content" = $2,
                "status" = 'SUCCEEDED',
                "updatedAt" = now()
            WHERE "id" = $1
            """,
            assistant_message_id,
            final_answer,
        )

        ref = str(repository["indexedCommitSha"] or repository["defaultBranch"])
        citations: list[dict[str, Any]] = []

        for index, document in enumerate(documents[:6], start=1):
            path = document.metadata.get("path")
            start_line = document.metadata.get("start_line")
            end_line = document.metadata.get("end_line")

            if (
                not isinstance(path, str)
                or not isinstance(start_line, int)
                or not isinstance(end_line, int)
            ):
                continue

            url = github_line_url(
                str(repository["htmlUrl"]) if repository["htmlUrl"] else None,
                ref,
                path,
                start_line,
                end_line,
            )
            citation_id = uuid.uuid4().hex
            await connection.execute(
                """
                INSERT INTO "ChatMessageCitation"
                  ("id", "messageId", "sourceIndex", "path", "startLine",
                   "endLine", "url")
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                citation_id,
                assistant_message_id,
                index,
                path,
                start_line,
                end_line,
                url,
            )
            citations.append(
                {
                    "id": citation_id,
                    "source_index": index,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "url": url,
                },
            )

        await connection.execute(
            """
            UPDATE "ChatConversation"
            SET "updatedAt" = now()
            WHERE "id" = $1
            """,
            conversation_id,
        )
        yield sse_event("citations", {"citations": citations})
        yield sse_event("done", {"message_id": assistant_message_id})
    except Exception as error:
        message = redacted_error(error)
        await connection.execute(
            """
            INSERT INTO "ChatMessage"
              ("id", "conversationId", "role", "content", "status",
               "errorMessage", "updatedAt")
            VALUES ($1, $2, 'ASSISTANT', '', 'FAILED', $3, now())
            ON CONFLICT ("id") DO UPDATE
            SET "status" = 'FAILED',
                "errorMessage" = $3,
                "updatedAt" = now()
            """,
            assistant_message_id,
            conversation_id,
            message,
        )
        yield sse_event("error", {"message": message})
