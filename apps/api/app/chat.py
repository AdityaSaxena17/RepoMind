from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from .config import get_settings
from .retrieval import RepoMindParentHybridRetriever, create_embedding_model


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


def build_chat_chain() -> Any:
    """Build the LCEL repository QA chain with hybrid retrieval context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are RepoMind, an AI assistant answering questions about one "
                "indexed GitHub repository. Use only the provided repository context "
                "and chat history. Cite code claims with source numbers like [1] or "
                "[2]. If the context does not contain enough evidence, say what is "
                "missing and suggest a precise follow-up search. Never invent file "
                "paths, APIs, line numbers, issues, pull requests, or behavior.",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Question:\n{question}\n\nRepository context:\n{context}",
            ),
        ],
    )
    retrieved_context = RunnableParallel(
        question=RunnablePassthrough(),
        documents=RunnableLambda(lambda value: value["documents"]),
    ) | RunnablePassthrough.assign(
        context=RunnableLambda(lambda value: format_documents(value["documents"])),
    )

    return (
        RunnableParallel(
            question=RunnableLambda(lambda value: value["question"]),
            history=RunnableLambda(lambda value: value["history"]),
            context=RunnablePassthrough() | retrieved_context
            | RunnableLambda(lambda value: value["context"]),
        )
        | prompt
        | create_chat_model()
        | StrOutputParser()
    )


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
        )
        documents = await retriever.ainvoke(str(user_message["content"]))

        if not documents:
            answer = (
                "I could not find enough indexed repository context to answer that. "
                "Try asking about a more specific file, function, error message, "
                "or symbol."
            )
            answer_parts.append(answer)
            yield sse_event("delta", {"text": answer})
        else:
            chain = build_chat_chain()
            history = await load_history(connection, conversation_id, message_id)

            async for delta in chain.astream(
                {
                    "question": str(user_message["content"]),
                    "history": history,
                    "documents": documents,
                },
            ):
                text = str(delta)
                answer_parts.append(text)
                yield sse_event("delta", {"text": text})

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
