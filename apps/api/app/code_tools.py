"""Indexed-code tools with owned database reads and request-scoped citations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

import asyncpg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import get_settings
from .embeddings import create_embedding_model
from .reranking import RERANKER_INSTRUCTION, RERANKER_MAX_LENGTH, RERANKER_REVISION
from .retrieval import (
    FINAL_PARENT_CONTEXTS,
    KEYWORD_CANDIDATES,
    PARENT_LINE_EXPANSION,
    PARENT_MAX_CHARS,
    PARENT_MAX_LINES,
    RRF_K,
    VECTOR_CANDIDATES,
    RepoMindParentHybridRetriever,
)
from .tool_errors import AuthorizationError, ScopeViolationError
from .tool_registry import RequestScope, ToolPolicy, ToolRegistry

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
logger = logging.getLogger(__name__)
CACHE_MAX_BYTES = 256 * 1024
CACHE_TIMEOUT_SECONDS = 0.25


class _CachedWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=PARENT_MAX_CHARS)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exact_range(self) -> _CachedWindow:
        if self.end_line - self.start_line + 1 != len(self.content.split("\n")):
            raise ValueError("Cached text must match its evidence range.")
        if self.end_line - self.start_line + 1 > PARENT_MAX_LINES:
            raise ValueError("Cached evidence exceeds the context line limit.")
        return self


class _CachedRead(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    identity: str
    windows: list[_CachedWindow] = Field(default_factory=list)
    missing_information: list[Nonempty] = Field(default_factory=list)
    context_notes: list[Nonempty] = Field(default_factory=list)
    file_line_count: int | None = Field(default=None, ge=0)


class CodeEvidence(BaseModel):
    """Exact visible source window, pinned to the investigation's indexed commit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    source_index: int = Field(ge=1)
    corpus_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str = Field(min_length=1)
    url: str | None

    @model_validator(mode="after")
    def exact_range(self) -> CodeEvidence:
        if self.end_line - self.start_line + 1 != len(self.content.split("\n")):
            raise ValueError("Evidence text must match its declared line range.")
        return self


class SearchCodeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: Nonempty


class FileContextArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def bounded_range(self) -> FileContextArguments:
        if self.end_line is not None and (
            self.end_line < self.start_line
            or self.end_line - self.start_line + 1 > PARENT_MAX_LINES
        ):
            raise ValueError(
                f"Request an ordered range of at most {PARENT_MAX_LINES} lines."
            )
        return self


class CodeTools:
    """Bind tools to one request and own each connection acquisition or lock."""

    def __init__(
        self,
        registry: ToolRegistry,
        embeddings: Embeddings | None,
        commit_sha: str,
        *,
        pool: asyncpg.Pool | None = None,
        connection: asyncpg.Connection | None = None,
        rerank: bool | None = None,
        vector_k: int = VECTOR_CANDIDATES,
        keyword_k: int = KEYWORD_CANDIDATES,
        redis: Redis | None = None,
        last_indexed_at: datetime | None = None,
        cache_database_url: str | None = None,
    ) -> None:
        if (pool is None) == (connection is None):
            raise ValueError("Supply a pool or an exclusively owned connection.")
        if len(commit_sha) != 40 or any(
            c not in "0123456789abcdef" for c in commit_sha
        ):
            raise ValueError("Code investigations require an indexed commit SHA.")
        self.request = registry.request
        self.commit_sha = commit_sha
        self.pool, self.connection = pool, connection
        self.embeddings = embeddings
        self.rerank = get_settings().repository_rerank if rerank is None else rerank
        self.vector_k, self.keyword_k = vector_k, keyword_k
        self.redis, self.last_indexed_at = redis, last_indexed_at
        self.cache_database_url = cache_database_url or get_settings().database_url
        self.cache_stats = {"hits": 0, "misses": 0, "errors": 0}
        self._cache_unavailable = False
        self._connection_lock = asyncio.Lock()
        self.evidence: dict[str, CodeEvidence] = {}
        self.first_pass: list[CodeEvidence] = []
        self.first_query: str | None = None
        self._search_started = False
        policy = ToolPolicy(capability="code.read", repository_scoped=True)
        metadata = {
            "readOnlyHint": True,
            "destructiveHint": False,
            "_meta": {"repomind": policy.model_dump()},
        }
        for name, coroutine, schema, description in (
            (
                "search_repository_code",
                self.search_repository_code,
                SearchCodeArguments,
                "Search the indexed commit with hybrid retrieval. Returns ranked "
                "code windows with exact paths, line ranges and stable citation "
                "IDs/numbers. Preserve the requested behavior and explicit identifiers "
                "or paths in the query rather than replacing them with broad "
                "topic words. "
                "Refine the query only for unresolved requested facts; "
                "use get_file_context to inspect a returned path more closely. "
                "Live PR contents are not part of this snapshot.",
            ),
            (
                "get_file_context",
                self.get_file_context,
                FileContextArguments,
                "Read an exact repository-relative path from the indexed commit. "
                "Use a path observed in search results or supplied in the question. "
                "Line numbers are inclusive and start at 1. Source IDs/numbers "
                "remain stable for identical evidence windows. "
                "context_notes describe optional unread context; missing_information "
                "identifies an incomplete requested read. "
                f"Return at most {PARENT_MAX_LINES} lines and "
                f"{PARENT_MAX_CHARS} characters.",
            ),
        ):
            registry.register_local(
                StructuredTool(
                    name=name,
                    description=description,
                    args_schema=schema,
                    coroutine=coroutine,
                    metadata=metadata,
                    response_format="content_and_artifact",
                    handle_tool_error=True,
                    handle_validation_error=(
                        "Invalid code-tool arguments: query/path must be nonempty "
                        "strings; line numbers must be integers >= 1, end_line >= "
                        "start_line, and a range must contain at most "
                        f"{PARENT_MAX_LINES} "
                        "lines. Split larger reads into separate ranges. Only supply "
                        "fields listed in the tool schema."
                    ),
                ),
                policy,
                self.authorize,
            )

    @asynccontextmanager
    async def _read(
        self, scope: RequestScope
    ) -> AsyncIterator[tuple[asyncpg.Connection, str | None]]:
        if scope != self.request:
            raise ScopeViolationError("Code tool request identity does not match.")
        async with AsyncExitStack() as stack:
            if self.pool is not None:
                connection = await stack.enter_async_context(
                    self.pool.acquire(timeout=10)
                )
            else:
                await stack.enter_async_context(self._connection_lock)
                connection = self.connection
            if connection is None:
                raise RuntimeError("Code tool connection is unavailable.")
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                repository = await connection.fetchrow(
                    """
                    SELECT r."indexedCommitSha", r."indexStatus", r."htmlUrl",
                           r."lastIndexedAt"
                    FROM "Repository" r
                    JOIN "UserRepository" ur ON ur."repositoryId" = r."id"
                    WHERE r."id" = $1 AND ur."userId" = $2
                      AND r."accessStatus" = 'ACTIVE'
                      AND ur."accessStatus" = 'ACTIVE'
                    """,
                    scope.repository_id,
                    scope.user_id,
                )
                if not repository:
                    raise AuthorizationError("Repository access is no longer active.")
                if (
                    repository["indexStatus"] != "READY"
                    or repository["indexedCommitSha"] != self.commit_sha
                    or (
                        self.last_indexed_at is not None
                        and repository["lastIndexedAt"] != self.last_indexed_at
                    )
                ):
                    raise ToolException(
                        "Indexed snapshot changed or is not ready; "
                        "restart the investigation."
                    )
                html_url = repository["htmlUrl"]
                yield connection, str(html_url) if html_url else None

    async def authorize(self, scope: RequestScope) -> None:
        """Check access before invocation and again within the execution snapshot."""
        async with self._read(scope):
            pass

    def _cache_key(self, name: str, arguments: dict[str, JsonValue]) -> str | None:
        settings = get_settings()
        if (
            self.redis is None
            or self.last_indexed_at is None
            or self._cache_unavailable
            or settings.code_context_cache_ttl_seconds == 0
        ):
            return None
        database = urlsplit(self.cache_database_url)
        identity = {
            "database": [database.hostname, database.port, database.path],
            "repository": self.request.repository_id,
            "commit": self.commit_sha,
            "index_revision": self.last_indexed_at.isoformat(),
            "name": name,
            "arguments": arguments,
            "embedding_profile": settings.embedding_profile,
            "retrieval": [
                self.vector_k,
                self.keyword_k,
                FINAL_PARENT_CONTEXTS,
                RRF_K,
                PARENT_LINE_EXPANSION,
                PARENT_MAX_LINES,
                PARENT_MAX_CHARS,
                self.rerank,
            ],
            "reranker": [RERANKER_REVISION, RERANKER_INSTRUCTION, RERANKER_MAX_LENGTH],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return f"repomind:code-context:v4:{digest}"

    async def _cached_read(self, key: str | None) -> _CachedRead | None:
        if key is None or self.redis is None:
            return None
        try:
            raw = await asyncio.wait_for(self.redis.get(key), CACHE_TIMEOUT_SECONDS)
            if raw is not None:
                if len(raw) > CACHE_MAX_BYTES:
                    raise ValueError("Oversized cache entry.")
                entry = _CachedRead.model_validate_json(raw)
                if entry.identity != key:
                    raise ValueError("Cache identity mismatch.")
                self.cache_stats["hits"] += 1
                return entry
        except (RedisError, TimeoutError, asyncio.TimeoutError, ValueError) as error:
            self.cache_stats["errors"] += 1
            self._cache_unavailable = isinstance(
                error, (RedisError, TimeoutError, asyncio.TimeoutError)
            )
            logger.warning("Code context cache read failed (%s).", type(error).__name__)
        self.cache_stats["misses"] += 1
        return None

    async def _cache_write(self, key: str | None, entry: _CachedRead) -> None:
        if key is None or self.redis is None or self._cache_unavailable:
            return
        raw = entry.model_dump_json().encode()
        if len(raw) > CACHE_MAX_BYTES:
            return
        try:
            await asyncio.wait_for(
                self.redis.set(
                    key, raw, ex=get_settings().code_context_cache_ttl_seconds
                ),
                CACHE_TIMEOUT_SECONDS,
            )
        except (RedisError, TimeoutError, asyncio.TimeoutError) as error:
            self.cache_stats["errors"] += 1
            self._cache_unavailable = True
            logger.warning(
                "Code context cache write failed (%s).", type(error).__name__
            )

    def _sources(
        self, documents: list[Document], html_url: str | None
    ) -> list[CodeEvidence]:
        sources = []
        for document in documents:
            if not document.page_content:
                continue
            path = document.metadata["path"]
            first, last = document.metadata["start_line"], document.metadata["end_line"]
            source_id = f"code:{self.commit_sha}:{path}:{first}:{last}"
            existing = self.evidence.get(source_id)
            if existing is not None:
                if existing.content != document.page_content:
                    raise ScopeViolationError(
                        "Indexed source changed without a new commit identity."
                    )
                sources.append(existing)
                continue
            source = CodeEvidence(
                id=source_id,
                source_index=len(self.evidence) + 1,
                corpus_commit_sha=self.commit_sha,
                path=path,
                start_line=first,
                end_line=last,
                content=document.page_content,
                url=(
                    f"{html_url.rstrip('/')}/blob/{self.commit_sha}/"
                    f"{quote(path, safe='/')}"
                    f"#L{first}-L{last}"
                    if html_url
                    else None
                ),
            )
            self.evidence[source_id] = source
            sources.append(source)
        return sources

    async def search_repository_code(
        self, query: str, config: RunnableConfig
    ) -> tuple[str, dict[str, Any]]:
        """Run existing hybrid retrieval under one authorized read snapshot."""
        first_pass = not self._search_started
        self._search_started = True
        if first_pass:
            self.first_query = query
        key = self._cache_key("search_repository_code", {"query": query})
        async with self._read(self.request) as (connection, html_url):
            entry = await self._cached_read(key)
            cache_hit = entry is not None
            if entry is None:
                if self.embeddings is None:
                    self.embeddings = create_embedding_model()
                retriever = RepoMindParentHybridRetriever(
                    connection=connection,
                    embeddings=self.embeddings,
                    repository_id=self.request.repository_id,
                    rerank=self.rerank,
                    vector_k=self.vector_k,
                    keyword_k=self.keyword_k,
                )
                documents = await retriever.ainvoke(query, config=config)
                entry = _CachedRead(
                    identity=key or "",
                    windows=[
                        _CachedWindow(
                            path=doc.metadata["path"],
                            start_line=doc.metadata["start_line"],
                            end_line=doc.metadata["end_line"],
                            content=doc.page_content,
                            metadata={
                                k: v
                                for k, v in doc.metadata.items()
                                if k
                                not in {
                                    "path",
                                    "start_line",
                                    "end_line",
                                    "source_index",
                                }
                            },
                        )
                        for doc in documents
                        if doc.page_content
                    ],
                )
        if not cache_hit:
            await self._cache_write(key, entry)
        documents = [
            Document(
                page_content=window.content,
                metadata={
                    **window.metadata,
                    "path": window.path,
                    "start_line": window.start_line,
                    "end_line": window.end_line,
                },
            )
            for window in entry.windows
        ]
        sources = self._sources(documents, html_url)
        if first_pass:
            self.first_pass = sources
        result = {
            "sources": [source.model_dump(mode="json") for source in sources],
            "missing_information": []
            if sources
            else ["No indexed code matched this query."],
            "context_notes": [],
        }
        return json.dumps(result, ensure_ascii=False), result

    async def get_file_context(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Read whole indexed lines, reporting the actual range and truncation."""
        arguments = FileContextArguments(
            path=path, start_line=start_line, end_line=end_line
        )
        key = self._cache_key("get_file_context", arguments.model_dump(mode="json"))
        async with self._read(self.request) as (connection, html_url):
            entry = await self._cached_read(key)
            if entry is not None and any(
                w.path != path or w.start_line != start_line for w in entry.windows
            ):
                self.cache_stats["hits"] -= 1
                self.cache_stats["errors"] += 1
                self.cache_stats["misses"] += 1
                logger.warning("Cached file context does not match its request.")
                entry = None
            if entry is not None:
                sources = self._sources(
                    [
                        Document(
                            page_content=w.content,
                            metadata={
                                "path": w.path,
                                "start_line": w.start_line,
                                "end_line": w.end_line,
                            },
                        )
                        for w in entry.windows
                    ],
                    html_url,
                )
                cached_result = {
                    "sources": [source.model_dump(mode="json") for source in sources],
                    "missing_information": entry.missing_information,
                    "context_notes": entry.context_notes,
                }
                if entry.file_line_count is not None:
                    cached_result["file_line_count"] = entry.file_line_count
                return json.dumps(cached_result, ensure_ascii=False), cached_result
            row = await connection.fetchrow(
                'SELECT "content" FROM "RepositoryFile" '
                'WHERE "repositoryId"=$1 AND "path"=$2',
                self.request.repository_id,
                arguments.path,
            )
            if row is None:
                raise ToolException("File is not present in the indexed repository.")
            lines = str(row["content"]).splitlines()
        if not lines:
            result: dict[str, Any] = {
                "sources": [],
                "missing_information": ["The indexed file is empty."],
                "context_notes": [],
            }
            await self._cache_write(
                key,
                _CachedRead(
                    identity=key or "",
                    missing_information=result["missing_information"],
                ),
            )
            return json.dumps(result), result
        if start_line > len(lines):
            raise ToolException(
                f"Start line exceeds the indexed file's {len(lines)} lines; "
                f"request start_line between 1 and {len(lines)}."
            )
        requested_end = end_line or min(len(lines), start_line + PARENT_MAX_LINES - 1)
        last = min(requested_end, len(lines))
        selected = lines[start_line - 1 : last]
        while selected and len("\n".join(selected)) > PARENT_MAX_CHARS:
            selected.pop()
            last -= 1
        if not selected:
            raise ToolException("A source line exceeds the context character limit.")
        sources = self._sources(
            [
                Document(
                    page_content="\n".join(selected),
                    metadata={
                        "path": arguments.path,
                        "start_line": start_line,
                        "end_line": last,
                    },
                )
            ],
            html_url,
        )
        missing: list[str] = []
        notes: list[str] = []
        if end_line is not None and last < min(requested_end, len(lines)):
            missing.append(
                f"Requested context ends at line {requested_end}; "
                f"only lines through {last} were returned."
            )
        if end_line is None and last < len(lines):
            notes.append(
                f"The file has {len(lines)} lines; "
                "request another range to inspect the remainder."
            )
        result = {
            "sources": [source.model_dump(mode="json") for source in sources],
            "missing_information": missing,
            "context_notes": notes,
            "file_line_count": len(lines),
        }
        await self._cache_write(
            key,
            _CachedRead(
                identity=key or "",
                windows=[
                    _CachedWindow(
                        path=arguments.path,
                        start_line=start_line,
                        end_line=last,
                        content="\n".join(selected),
                    )
                ],
                missing_information=missing,
                context_notes=notes,
                file_line_count=len(lines),
            ),
        )
        return json.dumps(result, ensure_ascii=False), result
