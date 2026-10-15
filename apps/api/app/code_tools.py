"""Indexed-code tools with owned database reads and request-scoped citations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any
from urllib.parse import quote

import asyncpg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .config import get_settings
from .retrieval import (
    KEYWORD_CANDIDATES,
    PARENT_MAX_CHARS,
    PARENT_MAX_LINES,
    VECTOR_CANDIDATES,
    RepoMindParentHybridRetriever,
)
from .tool_errors import AuthorizationError, ScopeViolationError
from .tool_registry import RequestScope, ToolPolicy, ToolRegistry

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


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
    path: Nonempty
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
        embeddings: Embeddings,
        commit_sha: str,
        *,
        pool: asyncpg.Pool | None = None,
        connection: asyncpg.Connection | None = None,
        rerank: bool | None = None,
        vector_k: int = VECTOR_CANDIDATES,
        keyword_k: int = KEYWORD_CANDIDATES,
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
                "Search the indexed repository with hybrid retrieval. Refine the query "
                "if evidence is missing. Source IDs and numbers remain stable.",
            ),
            (
                "get_file_context",
                self.get_file_context,
                FileContextArguments,
                "Read an exact repository-relative path from the indexed snapshot. "
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
                    handle_validation_error=True,
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
                    SELECT r."indexedCommitSha", r."indexStatus", r."htmlUrl"
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
        async with self._read(self.request) as (connection, html_url):
            retriever = RepoMindParentHybridRetriever(
                connection=connection,
                embeddings=self.embeddings,
                repository_id=self.request.repository_id,
                rerank=self.rerank,
                vector_k=self.vector_k,
                keyword_k=self.keyword_k,
            )
            documents = await retriever.ainvoke(query, config=config)
        sources = self._sources(documents, html_url)
        if first_pass:
            self.first_pass = sources
        result = {
            "sources": [source.model_dump(mode="json") for source in sources],
            "missing_information": []
            if sources
            else ["No indexed code matched this query."],
        }
        return json.dumps(result, ensure_ascii=False), result

    async def get_file_context(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Read whole indexed lines, reporting the actual range and truncation."""
        arguments = FileContextArguments(
            path=path, start_line=start_line, end_line=end_line
        )
        async with self._read(self.request) as (connection, html_url):
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
            }
            return json.dumps(result), result
        if start_line > len(lines):
            raise ToolException("Start line exceeds the indexed file's line count.")
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
        missing = []
        if last < requested_end:
            missing.append(
                f"Requested context ends at line {requested_end}; "
                f"only lines through {last} were returned."
            )
        elif end_line is None and last < len(lines):
            missing.append(
                f"The file has {len(lines)} lines; "
                "request another range to inspect the remainder."
            )
        result = {
            "sources": [source.model_dump(mode="json") for source in sources],
            "missing_information": missing,
            "file_line_count": len(lines),
        }
        return json.dumps(result, ensure_ascii=False), result
