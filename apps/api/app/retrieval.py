from __future__ import annotations

import asyncio
import math
import re
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
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from .config import get_settings
from .embeddings import create_embedding_model as create_embedding_model
from .observability import log_event, trace_operation
from .reranking import score_passages

VECTOR_CANDIDATES = 24
KEYWORD_CANDIDATES = 24
FINAL_PARENT_CONTEXTS = 6
PARENT_LINE_EXPANSION = 80
PARENT_MAX_LINES = 250
PARENT_MAX_CHARS = 16_000
RRF_K = 60


def _preserve_parent_context(
    original: list[Document], completed: list[Document]
) -> list[Document]:
    """Restore whole parents until completion retains all originally visible text."""
    result = list(completed)
    original_lines = {
        (str(doc.metadata["path"]), line)
        for doc in original
        if doc.page_content
        for line in range(doc.metadata["start_line"], doc.metadata["end_line"] + 1)
    }
    # Restoring one window can remove coverage that another shifted window needs.
    # Each iteration restores a different parent, so this takes at most final_k steps.
    while missing := original_lines - {
        (str(doc.metadata["path"]), line)
        for doc in result
        if doc.page_content
        for line in range(doc.metadata["start_line"], doc.metadata["end_line"] + 1)
    }:
        for index, doc in enumerate(original):
            if doc.page_content and any(
                (str(doc.metadata["path"]), line) in missing
                for line in range(
                    doc.metadata["start_line"], doc.metadata["end_line"] + 1
                )
            ):
                # Restore provenance as well as text; rejected merges add no sources.
                result[index] = doc
                break
    return result


async def verify_embedding_profile(
    connection: asyncpg.Connection,
    repository_id: str,
) -> None:
    """Reject mixed, legacy or stale vector spaces before embedding a query."""
    log_event("database", "verify_embedding_profile", "started")
    if await connection.fetchval(
        'SELECT EXISTS (SELECT 1 FROM "RepositoryChunk" '
        'WHERE "repositoryId"=$1 AND ("embeddingProfile" IS DISTINCT FROM $2 '
        'OR vector_dims("embedding") <> $3))',
        repository_id,
        get_settings().embedding_profile,
        get_settings().embedding_dimensions,
    ):
        raise ValueError(
            "Repository embedding model changed; reindex this repository before chat."
        )
    log_event("database", "verify_embedding_profile", "completed")


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

        settings = get_settings()
        if any(
            len(vector) != settings.embedding_dimensions
            or not all(math.isfinite(value) for value in vector)
            or math.hypot(*vector) == 0
            for vector in embeddings
        ):
            raise ValueError(
                "Index embeddings have incompatible dimensions or invalid values."
            )

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
                   "startLine", "endLine", "content", "contentSha", "embedding",
                   "embeddingProfile")
                VALUES
                  ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11)
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
                settings.embedding_profile,
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
        await verify_embedding_profile(self.connection, self.repository_id)
        settings = get_settings()
        dimensions = settings.embedding_dimensions
        with trace_operation("embedding", "query"):
            query_vector = await self.embeddings.aembed_query(query)
        vector = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
        log_event("database", "vector_search", "started")
        rows = await self.connection.fetch(
            f"""
            SELECT "path", "startLine", "endLine", "content", "contentSha",
                   "embedding"::vector({dimensions})
                       <=> $2::vector({dimensions}) AS distance
            FROM "RepositoryChunk"
            WHERE "repositoryId" = $1
              AND "embeddingProfile" = $4
              AND vector_dims("embedding") = {dimensions}
            ORDER BY "embedding"::vector({dimensions}) <=> $2::vector({dimensions})
            LIMIT $3
            """,
            self.repository_id,
            vector,
            k,
            settings.embedding_profile,
        )
        log_event("database", "vector_search", "completed", rows=len(rows))

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


class RepoMindParentHybridRetriever(BaseRetriever):
    """Retrieve child chunks with vector+keyword search, then return parent windows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: asyncpg.Connection
    embeddings: Embeddings
    repository_id: str
    vector_k: int = VECTOR_CANDIDATES
    keyword_k: int = KEYWORD_CANDIDATES
    final_k: int = FINAL_PARENT_CONTEXTS
    rerank: bool = False

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
        # asyncpg permits only one operation at a time on this shared connection.
        vector_rows = await self._vector_candidates(query) if self.vector_k > 0 else []
        keyword_rows = (
            await self._keyword_candidates(query) if self.keyword_k > 0 else []
        )
        fused = self._fuse_candidates(vector_rows, keyword_rows)
        documents = await self._parent_documents(
            fused if self.rerank else fused[: self.final_k * 3]
        )
        if self.rerank and documents:
            # Deduplicate visible windows, retaining all retrieval provenance.
            unique: dict[tuple[str, int, int, str], Document] = {}
            for document in documents:
                if not document.page_content:
                    continue
                key = (
                    str(document.metadata["path"]),
                    int(document.metadata["start_line"]),
                    int(document.metadata["end_line"]),
                    document.page_content,
                )
                existing = unique.get(key)
                if existing is None:
                    unique[key] = document
                else:
                    existing.metadata["sources"] = sorted(
                        set(existing.metadata["sources"])
                        | set(document.metadata["sources"])
                    )
                    existing.metadata["score"] += document.metadata["score"]
            documents = list(unique.values())
            if not documents:
                return []
            scores = await asyncio.to_thread(
                score_passages,
                query,
                [f"{doc.metadata['path']}\n{doc.page_content}" for doc in documents],
            )
            for document, score in zip(documents, scores, strict=True):
                document.metadata["rerank_score"] = score
            # Stable ties retain the parent order derived from RRF child order.
            documents.sort(key=lambda doc: -float(doc.metadata["rerank_score"]))
            documents = documents[: self.final_k]
            for index, document in enumerate(documents, start=1):
                document.metadata["source_index"] = index
        return documents

    async def _vector_candidates(self, query: str) -> list[dict[str, Any]]:
        await verify_embedding_profile(self.connection, self.repository_id)
        settings = get_settings()
        dimensions = settings.embedding_dimensions
        with trace_operation("embedding", "query"):
            query_vector = await self.embeddings.aembed_query(query)
        vector = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
        log_event("database", "vector_search", "started")
        rows = await self.connection.fetch(
            f"""
            SELECT "id", "fileId", "path", "startLine", "endLine",
                   "contentSha", "content",
                   "embedding"::vector({dimensions})
                       <=> $2::vector({dimensions}) AS distance
            FROM "RepositoryChunk"
            WHERE "repositoryId" = $1
              AND "embeddingProfile" = $4
              AND vector_dims("embedding") = {dimensions}
            ORDER BY "embedding"::vector({dimensions}) <=> $2::vector({dimensions})
            LIMIT $3
            """,
            self.repository_id,
            vector,
            self.vector_k,
            settings.embedding_profile,
        )
        log_event("database", "vector_search", "completed", rows=len(rows))

        return [
            {
                "id": str(row["id"]),
                "file_id": str(row["fileId"]),
                "path": str(row["path"]),
                "start_line": int(row["startLine"]),
                "end_line": int(row["endLine"]),
                "content_sha": str(row["contentSha"]),
                "content": str(row["content"]),
                "distance": float(row["distance"]),
                "source": "vector",
                "rank": index + 1,
            }
            for index, row in enumerate(rows)
        ]

    async def _keyword_candidates(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        # CamelCase also occurs in prose (GitHub, RepoMind); a boost requires
        # backticks, snake_case, or explicit call syntax instead.
        identifiers = sorted(
            {
                word
                for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
                if "_" in word
            }
            | set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", query))
            | set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\(", query))
        )
        identifier_patterns = [
            f"(^|[^[:alnum:]_]){word}([^[:alnum:]_]|$)" for word in identifiers
        ]
        paths = sorted(set(re.findall(r"[\w@./\[\]-]+\.[A-Za-z][A-Za-z0-9]*", query)))
        log_event("database", "keyword_search", "started")
        rows = await self.connection.fetch(
            """
            WITH terms AS (
              -- Use English only to recognize stopwords; the index is unstemmed.
              SELECT DISTINCT token FROM ts_debug('english', $2)
              WHERE cardinality(lexemes) > 0
              UNION SELECT unnest($6::text[])
            ), search_query AS (
              SELECT string_agg(plainto_tsquery('simple', token)::text,
                                ' | ' ORDER BY token)::tsquery AS query
              FROM terms WHERE numnode(plainto_tsquery('simple', token)) > 0
            )
            SELECT c."id", c."fileId", c."path", c."startLine", c."endLine",
                   c."contentSha", c."content",
                   COALESCE(ts_rank_cd(to_tsvector('simple', c."content"),
                                       search_query.query), 0) AS rank_score,
                   (SELECT count(DISTINCT plainto_tsquery('simple', token)::text)
                    FROM terms WHERE to_tsvector('simple', c."content")
                                     @@ plainto_tsquery('simple', token))
                     AS matched_terms,
                   (EXISTS (SELECT 1 FROM unnest($4::text[]) pattern
                            WHERE c."content" ~ pattern)
                    OR EXISTS (SELECT 1 FROM unnest($5::text[]) path
                               WHERE c."path" = path
                                  OR right(c."path", length(path) + 1) = '/' || path))
                     AS exact_match
            FROM "RepositoryChunk" c, search_query
            WHERE c."repositoryId" = $1
              AND (to_tsvector('simple', c."content") @@ search_query.query
                   OR EXISTS (SELECT 1 FROM unnest($5::text[]) path
                              WHERE c."path" = path
                                 OR right(c."path", length(path) + 1) = '/' || path))
            -- Broad term coverage outranks repetition of a single generic word.
            ORDER BY exact_match DESC, matched_terms DESC, rank_score DESC,
                     c."path" COLLATE "C", c."chunkIndex", c."id"
            LIMIT $3
            """,
            self.repository_id,
            query,
            self.keyword_k,
            identifier_patterns,
            paths,
            identifiers,
        )
        log_event("database", "keyword_search", "completed", rows=len(rows))

        return [
            {
                "id": str(row["id"]),
                "file_id": str(row["fileId"]),
                "path": str(row["path"]),
                "start_line": int(row["startLine"]),
                "end_line": int(row["endLine"]),
                "content_sha": str(row["contentSha"]),
                "keyword_score": float(row["rank_score"]),
                "content": str(row["content"]),
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
        results.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["path"]),
                int(item["start_line"]),
                str(item["id"]),
            )
        )
        return results

    async def _parent_documents(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[Document]:
        if not candidates:
            return []

        file_ids = sorted({str(candidate["file_id"]) for candidate in candidates})
        log_event("database", "parent_windows", "started")
        rows = await self.connection.fetch(
            """
            SELECT "id", "path", "content", "lineCount"
            FROM "RepositoryFile"
            WHERE "repositoryId" = $1 AND "id" = ANY($2::text[])
            """,
            self.repository_id,
            file_ids,
        )
        log_event("database", "parent_windows", "completed", rows=len(rows))
        files = {str(row["id"]): row for row in rows}
        windows: list[dict[str, Any]] = []
        original: list[Document] | None = None

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

            if existing and original is not None:
                protected_end = max(existing["child_end"], int(candidate["end_line"]))
                prefix_start = min(int(existing["start_line"]), start_line)
                lines = str(file_row["content"]).splitlines()
                if (
                    len("\n".join(lines[prefix_start - 1 : protected_end]))
                    > PARENT_MAX_CHARS
                ):
                    existing = None

            if existing:
                existing["start_line"] = min(int(existing["start_line"]), start_line)
                existing["end_line"] = max(int(existing["end_line"]), end_line)
                existing["score"] = float(existing["score"]) + float(candidate["score"])
                existing["sources"].update(candidate["sources"])
                existing["child_start"] = min(
                    existing["child_start"], int(candidate["start_line"])
                )
                existing["child_end"] = max(
                    existing["child_end"], int(candidate["end_line"])
                )
                continue

            if original is not None:
                # Capacity fixes parent identities/order. Later children may use
                # their padding budget, but cannot evict ranked child evidence.
                for selected in windows:
                    if selected["file_id"] != candidate["file_id"]:
                        continue
                    child_start = min(
                        selected["child_start"], int(candidate["start_line"])
                    )
                    child_end = max(selected["child_end"], int(candidate["end_line"]))
                    if child_end - child_start + 1 > PARENT_MAX_LINES:
                        continue
                    left = min(
                        int(selected["start_line"]), int(candidate["start_line"])
                    )
                    right = max(int(selected["end_line"]), int(candidate["end_line"]))
                    if right - left + 1 > PARENT_MAX_LINES:
                        left = max(left, child_end - PARENT_MAX_LINES + 1)
                        right = min(right, left + PARENT_MAX_LINES - 1)
                    lines = str(file_row["content"]).splitlines()
                    if len("\n".join(lines[left - 1 : child_end])) > PARENT_MAX_CHARS:
                        continue
                    selected.update(
                        start_line=left,
                        end_line=right,
                        child_start=child_start,
                        child_end=child_end,
                    )
                    selected["score"] += candidate["score"]
                    selected["sources"].update(candidate["sources"])
                    break
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
                    "child_start": int(candidate["start_line"]),
                    "child_end": int(candidate["end_line"]),
                },
            )

            if not self.rerank and len(windows) >= self.final_k:
                # Render once at capacity to freeze actual text after character
                # clipping, not just declared ranges. Files are already loaded.
                original = self._render_parent_documents(windows[: self.final_k])

        selected_windows = windows if self.rerank else windows[: self.final_k]
        documents = self._render_parent_documents(selected_windows)
        return (
            _preserve_parent_context(original, documents)
            if original is not None
            else documents
        )

    def _render_parent_documents(self, windows: list[dict[str, Any]]) -> list[Document]:
        """Render bounded parent text and independent provenance from loaded files."""
        documents: list[Document] = []

        for index, window in enumerate(windows):
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
