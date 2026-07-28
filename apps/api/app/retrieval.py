from __future__ import annotations

import asyncio
import uuid
from typing import Any

import asyncpg
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict

from .config import get_settings

EMBEDDING_DIMENSIONS = 1536
VECTOR_CANDIDATES = 12
KEYWORD_CANDIDATES = 12
FINAL_PARENT_CONTEXTS = 6
PARENT_LINE_EXPANSION = 80
PARENT_MAX_LINES = 250
PARENT_MAX_CHARS = 16_000
RRF_K = 60


def create_embedding_model() -> OpenAIEmbeddings:
    """Create the shared LangChain embeddings model for indexing and retrieval."""
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for repository indexing.")

    return OpenAIEmbeddings(
        model=settings.openai_embeddings_model,
        dimensions=EMBEDDING_DIMENSIONS,
        chunk_size=64,
    )


class RepoMindRepositoryVectorStore(VectorStore):
    """LangChain vector store backed by RepoMind's existing RepositoryChunk table."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        embeddings: Embeddings,
        repository_id: str,
        file_ids_by_path: dict[str, str] | None = None,
    ) -> None:
        self.connection = connection
        self._embeddings = embeddings
        self.repository_id = repository_id
        self.file_ids_by_path = file_ids_by_path or {}

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding model used by this vector store."""
        return self._embeddings

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        connection: asyncpg.Connection | None = None,
        repository_id: str | None = None,
        **kwargs: Any,
    ) -> RepoMindRepositoryVectorStore:
        """Build a store from texts only when the target repo connection is explicit."""
        del texts, metadatas, kwargs

        if connection is None or repository_id is None:
            raise ValueError("connection and repository_id are required.")

        return cls(connection, embedding, repository_id)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Use async writes so indexing does not block the worker event loop."""
        del documents, kwargs
        raise NotImplementedError("Use aadd_documents or awrite_embedded_documents.")

    async def aembed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Embed LangChain documents with the configured embedding model."""
        vectors = await self.embeddings.aembed_documents(
            [document.page_content for document in documents],
        )
        return [[float(value) for value in vector] for vector in vectors]

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Embed and insert documents into RepositoryChunk."""
        vectors = await self.aembed_documents(documents)
        return await self.awrite_embedded_documents(documents, vectors, **kwargs)

    async def awrite_embedded_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        **kwargs: Any,
    ) -> list[str]:
        """Insert already-embedded documents into RepositoryChunk."""
        ids = kwargs.get("ids")

        if ids is not None and not isinstance(ids, list):
            raise TypeError("ids must be a list when provided.")

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have matching lengths.")

        inserted_ids: list[str] = []

        for index, document in enumerate(documents):
            path = document.metadata.get("path")
            start_line = document.metadata.get("start_line")
            end_line = document.metadata.get("end_line")
            content_sha = document.metadata.get("content_sha")

            if (
                not isinstance(path, str)
                or not isinstance(start_line, int)
                or not isinstance(end_line, int)
                or not isinstance(content_sha, str)
            ):
                raise ValueError("chunk documents require path and line metadata.")

            file_id = self.file_ids_by_path.get(path)

            if file_id is None:
                raise ValueError(f"file id is missing for {path}.")

            document_id = (
                ids[index]
                if (
                    isinstance(ids, list)
                    and index < len(ids)
                    and isinstance(ids[index], str)
                )
                else uuid.uuid4().hex
            )
            vector = "[" + ",".join(str(value) for value in embeddings[index]) + "]"

            await self.connection.execute(
                """
                INSERT INTO "RepositoryChunk"
                  ("id", "repositoryId", "fileId", "chunkIndex", "path",
                   "startLine", "endLine", "content", "contentSha", "embedding")
                VALUES
                  ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
                """,
                document_id,
                self.repository_id,
                file_id,
                index,
                path,
                start_line,
                end_line,
                document.page_content,
                content_sha,
                vector,
            )
            inserted_ids.append(document_id)

        return inserted_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Prefer async retrieval from FastAPI/worker code paths."""
        del query, k, kwargs
        raise NotImplementedError("Use asimilarity_search.")

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Return similar repository chunks as LangChain documents."""
        del kwargs
        query_vector = (await self.embeddings.aembed_query(query))
        vector = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
        rows = await self.connection.fetch(
            """
            SELECT "path", "startLine", "endLine", "content", "contentSha",
                   "embedding" <=> $2::vector AS distance
            FROM "RepositoryChunk"
            WHERE "repositoryId" = $1
            ORDER BY "embedding" <=> $2::vector
            LIMIT $3
            """,
            self.repository_id,
            vector,
            k,
        )

        return [
            Document(
                page_content=str(row["content"]),
                metadata={
                    "repository_id": self.repository_id,
                    "path": str(row["path"]),
                    "start_line": int(row["startLine"]),
                    "end_line": int(row["endLine"]),
                    "content_sha": str(row["contentSha"]),
                    "distance": float(row["distance"]),
                },
            )
            for row in rows
        ]


class RepoMindRepositoryRetriever(BaseRetriever):
    """LangChain retriever for repository chunks stored in pgvector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: RepoMindRepositoryVectorStore
    k: int = 6

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        raise NotImplementedError("Use async retriever calls with ainvoke.")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        return await self.vector_store.asimilarity_search(query, k=self.k)


class RepoMindParentHybridRetriever(BaseRetriever):
    """Retrieve child chunks with vector+keyword search, then return parent windows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: asyncpg.Connection
    embeddings: Embeddings
    repository_id: str
    vector_k: int = VECTOR_CANDIDATES
    keyword_k: int = KEYWORD_CANDIDATES
    final_k: int = FINAL_PARENT_CONTEXTS

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        raise NotImplementedError("Use async retriever calls with ainvoke.")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        vector_rows, keyword_rows = await asyncio.gather(
            self._vector_candidates(query),
            self._keyword_candidates(query),
        )
        fused = self._fuse_candidates(vector_rows, keyword_rows)
        return await self._parent_documents(fused[: self.final_k * 3])

    async def _vector_candidates(self, query: str) -> list[dict[str, Any]]:
        query_vector = await self.embeddings.aembed_query(query)
        vector = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
        rows = await self.connection.fetch(
            """
            SELECT "id", "fileId", "path", "startLine", "endLine", "contentSha",
                   "embedding" <=> $2::vector AS distance
            FROM "RepositoryChunk"
            WHERE "repositoryId" = $1
            ORDER BY "embedding" <=> $2::vector
            LIMIT $3
            """,
            self.repository_id,
            vector,
            self.vector_k,
        )

        return [
            {
                "id": str(row["id"]),
                "file_id": str(row["fileId"]),
                "path": str(row["path"]),
                "start_line": int(row["startLine"]),
                "end_line": int(row["endLine"]),
                "content_sha": str(row["contentSha"]),
                "distance": float(row["distance"]),
                "source": "vector",
                "rank": index + 1,
            }
            for index, row in enumerate(rows)
        ]

    async def _keyword_candidates(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        rows = await self.connection.fetch(
            """
            WITH search_query AS (
              SELECT websearch_to_tsquery('simple', $2) AS query
            )
            SELECT c."id", c."fileId", c."path", c."startLine", c."endLine",
                   c."contentSha",
                   ts_rank_cd(to_tsvector('simple', c."content"), search_query.query)
                     AS rank_score
            FROM "RepositoryChunk" c, search_query
            WHERE c."repositoryId" = $1
              AND to_tsvector('simple', c."content") @@ search_query.query
            ORDER BY rank_score DESC
            LIMIT $3
            """,
            self.repository_id,
            query,
            self.keyword_k,
        )

        return [
            {
                "id": str(row["id"]),
                "file_id": str(row["fileId"]),
                "path": str(row["path"]),
                "start_line": int(row["startLine"]),
                "end_line": int(row["endLine"]),
                "content_sha": str(row["contentSha"]),
                "keyword_score": float(row["rank_score"]),
                "source": "keyword",
                "rank": index + 1,
            }
            for index, row in enumerate(rows)
        ]

    @staticmethod
    def _fuse_candidates(
        vector_rows: list[dict[str, Any]],
        keyword_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for row in [*vector_rows, *keyword_rows]:
            chunk_id = str(row["id"])
            item = fused.setdefault(
                chunk_id,
                {
                    **row,
                    "score": 0.0,
                    "sources": set[str](),
                },
            )
            item["score"] = float(item["score"]) + 1 / (RRF_K + int(row["rank"]))
            item["sources"].add(str(row["source"]))

        results = list(fused.values())
        results.sort(key=lambda item: float(item["score"]), reverse=True)
        return results

    async def _parent_documents(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[Document]:
        if not candidates:
            return []

        file_ids = sorted({str(candidate["file_id"]) for candidate in candidates})
        rows = await self.connection.fetch(
            """
            SELECT "id", "path", "content", "lineCount"
            FROM "RepositoryFile"
            WHERE "repositoryId" = $1 AND "id" = ANY($2::text[])
            """,
            self.repository_id,
            file_ids,
        )
        files = {str(row["id"]): row for row in rows}
        windows: list[dict[str, Any]] = []

        for candidate in candidates:
            file_row = files.get(str(candidate["file_id"]))

            if file_row is None:
                continue

            line_count = int(file_row["lineCount"])
            start_line = max(1, int(candidate["start_line"]) - PARENT_LINE_EXPANSION)
            end_line = min(
                line_count,
                int(candidate["end_line"]) + PARENT_LINE_EXPANSION,
            )

            if end_line - start_line + 1 > PARENT_MAX_LINES:
                center = (
                    int(candidate["start_line"]) + int(candidate["end_line"])
                ) // 2
                start_line = max(1, center - (PARENT_MAX_LINES // 2))
                end_line = min(line_count, start_line + PARENT_MAX_LINES - 1)
                start_line = max(1, end_line - PARENT_MAX_LINES + 1)

            existing = next(
                (
                    window
                    for window in windows
                    if window["file_id"] == candidate["file_id"]
                    and start_line <= int(window["end_line"]) + 20
                    and end_line >= int(window["start_line"]) - 20
                    and max(end_line, int(window["end_line"]))
                    - min(start_line, int(window["start_line"]))
                    + 1
                    <= PARENT_MAX_LINES
                ),
                None,
            )

            if existing:
                existing["start_line"] = min(int(existing["start_line"]), start_line)
                existing["end_line"] = max(int(existing["end_line"]), end_line)
                existing["score"] = float(existing["score"]) + float(candidate["score"])
                existing["sources"].update(candidate["sources"])
                continue

            windows.append(
                {
                    "file_id": str(candidate["file_id"]),
                    "path": str(file_row["path"]),
                    "content": str(file_row["content"]),
                    "start_line": start_line,
                    "end_line": end_line,
                    "score": float(candidate["score"]),
                    "sources": set(candidate["sources"]),
                },
            )

            if len(windows) >= self.final_k:
                break

        documents: list[Document] = []

        for index, window in enumerate(windows[: self.final_k]):
            lines = str(window["content"]).splitlines()
            start_line = int(window["start_line"])
            end_line = int(window["end_line"])
            selected_lines = lines[start_line - 1 : end_line]
            page_content = "\n".join(selected_lines)

            while len(page_content) > PARENT_MAX_CHARS and selected_lines:
                selected_lines.pop()
                end_line -= 1
                page_content = "\n".join(selected_lines)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "repository_id": self.repository_id,
                        "source_index": index + 1,
                        "path": str(window["path"]),
                        "start_line": start_line,
                        "end_line": max(start_line, end_line),
                        "score": float(window["score"]),
                        "sources": sorted(str(source) for source in window["sources"]),
                    },
                ),
            )

        return documents


def build_repository_context_runnable(
    retriever: BaseRetriever,
) -> Any:
    """Compose query, retrieved docs, and formatted context for future RAG chains."""

    def format_context(value: dict[str, Any]) -> str:
        documents = value.get("documents")

        if not isinstance(documents, list):
            return ""

        sections: list[str] = []

        for document in documents:
            if not isinstance(document, Document):
                continue

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
            sections.append(f"{location}\n{document.page_content}")

        return "\n\n".join(sections)

    return RunnableParallel(
        query=RunnablePassthrough(),
        documents=retriever,
    ) | RunnablePassthrough.assign(context=RunnableLambda(format_context))
