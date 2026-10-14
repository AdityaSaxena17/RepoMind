"""Frozen-corpus preparation and the current production-agent evaluation adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

import anyio
import asyncpg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from ..agents import AgentFindings, AgentLimits, investigate_code, investigate_github
from ..chat import stream_repository_answer
from ..code_tools import CodeTools
from ..config import get_settings, normalize_asyncpg_database_url
from ..github_mcp_server import mcp, search_query, validate_page
from ..retrieval import (
    KEYWORD_CANDIDATES,
    VECTOR_CANDIDATES,
    RepoMindParentHybridRetriever,
    create_embedding_model,
)
from ..tool_registry import RequestScope, ToolRegistry, parse_tool_policy
from ..worker import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    chunk_file,
    promote_index,
    should_skip_path,
)
from .models import (
    CodeContext,
    Dataset,
    EvaluationResult,
    Fixtures,
    Metric,
    Question,
    ToolExecution,
)
from .scoring import build_catalog, ratio, unit_overlap, visible_lines

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).parent / "data"
ARTIFACTS = ROOT / ".eval-artifacts"
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
VECTOR_LIST = TypeAdapter(list[list[float]])


def digest(value: str) -> str:
    """Return a stable content hash used for corpus, cache, and prompt identities."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class Benchmark:
    dataset: Dataset
    fixtures: Fixtures
    corpus: dict[str, str]
    blobs: dict[str, str]


def load_benchmark(
    dataset_path: Path = DATA / "dataset.json",
    fixtures_path: Path = DATA / "github.json",
    corpus_root: Path = ROOT,
) -> Benchmark:
    """Load and verify every annotation against tracked files at the pinned commit."""
    dataset = Dataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    fixtures = Fixtures.model_validate_json(fixtures_path.read_text(encoding="utf-8"))
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "-z", dataset.corpus_commit_sha],
        cwd=corpus_root,
        check=True,
        capture_output=True,
    ).stdout
    corpus: dict[str, str] = {}
    blobs: dict[str, str] = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        _mode, kind, blob = header.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if kind != "blob" or should_skip_path(path, None):
            continue
        raw = subprocess.run(
            ["git", "show", f"{dataset.corpus_commit_sha}:{path}"],
            cwd=corpus_root,
            check=True,
            capture_output=True,
        ).stdout
        if should_skip_path(path, len(raw)):
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue  # Production indexing also excludes non-UTF-8 blobs.
        if "-----BEGIN " in content[:2000]:
            continue
        corpus[path], blobs[path] = content, blob
    if not corpus:
        raise ValueError("The frozen corpus contains no indexable files.")
    for question in dataset.questions:
        if question.fixture_scenario not in fixtures.scenarios:
            raise ValueError(f"Unknown fixture scenario in {question.id}.")
        calls = fixtures.scenarios[question.fixture_scenario]
        call_ids = {call.id for call in calls}
        if len(call_ids) != len(calls):
            raise ValueError(f"Duplicate fixture ID in {question.id}.")
        if any(set(f.fixture_evidence) - call_ids for f in question.reference_facts):
            raise ValueError(f"Unknown fixture evidence in {question.id}.")
        for unit in question.required_evidence + question.relevant_evidence:
            for alternative in unit.alternatives:
                for span in alternative:
                    source = corpus.get(span.path)
                    if source is None or digest(source) != span.file_sha256:
                        raise ValueError(f"Annotation source changed: {question.id}.")
                    excerpt = "\n".join(
                        source.splitlines()[span.start_line - 1 : span.end_line],
                    )
                    if span.end_line > len(source.splitlines()):
                        raise ValueError(f"Annotation exceeds source: {question.id}.")
                    if excerpt != span.excerpt:
                        raise ValueError(f"Annotation range changed: {question.id}.")
    return Benchmark(dataset, fixtures, corpus, blobs)


class CachedEmbeddings(Embeddings):
    """Cache validated vectors; only cache misses invoke the configured provider."""

    def __init__(self, cache: Path = ARTIFACTS / "embeddings") -> None:
        self.cache = cache
        settings = get_settings()
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.profile = settings.embedding_profile
        # Preserve historical OpenAI cache entries; Qwen also pins weights and prompt.
        self.cache_model = (
            self.model if settings.embeddings_provider == "openai" else self.profile
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Require asynchronous embedding calls in evaluation execution."""
        raise NotImplementedError("Use aembed_documents.")

    def embed_query(self, text: str) -> list[float]:
        """Require asynchronous embedding calls in evaluation execution."""
        raise NotImplementedError("Use aembed_query.")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with cache identities distinct from query embeddings."""
        return await self._embed(texts, "document")

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query, reusing a cached vector when available."""
        return (await self._embed([text], "query"))[0]

    async def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        paths = [
            self.cache
            / (digest(f"{self.cache_model}:{self.dimensions}:{kind}:{text}") + ".json")
            for text in texts
        ]
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, path in enumerate(paths):
            if path.exists():
                cached = VECTOR_LIST.validate_json(path.read_text(encoding="utf-8"))
                if len(cached) != 1:
                    raise ValueError("Invalid embedding cache entry.")
                vectors[index] = cached[0]
            else:
                missing.append(index)
        if missing:
            provider = create_embedding_model()
            if kind == "query":
                generated = [
                    await asyncio.wait_for(
                        provider.aembed_query(texts[missing[0]]),
                        timeout=120,
                    )
                ]
            else:
                generated = await asyncio.wait_for(
                    provider.aembed_documents([texts[i] for i in missing]),
                    timeout=300,
                )
            if len(generated) != len(missing):
                raise ValueError("Embedding provider returned the wrong vector count.")
            for index, vector in zip(missing, generated, strict=True):
                vectors[index] = vector
        for vector in vectors.values():
            if (
                len(vector) != self.dimensions
                or not all(map(math.isfinite, vector))
                or math.hypot(*vector) == 0
            ):
                raise ValueError("Invalid embedding dimensions or non-finite values.")
        if missing:
            self.cache.mkdir(parents=True, exist_ok=True)
            for index in missing:
                # Independent experiments may fill the same cache entry concurrently.
                temporary = paths[index].with_suffix(f".{uuid.uuid4().hex}.tmp")
                temporary.write_text(json.dumps([vectors[index]]), encoding="utf-8")
                temporary.replace(paths[index])
        return [vectors[index] for index in range(len(texts))]


async def connect_evaluation_database() -> asyncpg.Connection:
    """Connect only to the dedicated database using an unprivileged, separate role."""
    url = os.getenv("EVAL_DATABASE_URL")
    if not url:
        raise ValueError(
            "EVAL_DATABASE_URL is required; application URLs are never used."
        )
    parsed = urlsplit(url)
    app = urlsplit(get_settings().database_url)
    if (
        unquote(parsed.path.lstrip("/")) != "repomind_eval"
        or unquote(parsed.username or "") != "repomind_eval"
        or unquote(app.path.lstrip("/")) == "repomind_eval"
        or parsed.username == app.username
    ):
        raise ValueError("Use the separate repomind_eval database and role.")
    connection = await asyncpg.connect(
        normalize_asyncpg_database_url(url),
        timeout=10,
        command_timeout=60,
    )
    try:
        row = await connection.fetchrow(
            """SELECT current_database() AS db, current_user AS role,
                      rolsuper, rolcreatedb, rolcreaterole,
                      EXISTS (SELECT 1 FROM pg_auth_members WHERE member = r.oid)
                        AS memberships
               FROM pg_roles r WHERE rolname = current_user""",
        )
        if (
            not row
            or row["db"] != "repomind_eval"
            or row["role"] != "repomind_eval"
            or any(
                row[key]
                for key in (
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "memberships",
                )
            )
        ):
            raise ValueError(
                "Evaluation database identity or role privileges are unsafe."
            )
    except BaseException:
        await connection.close()
        raise
    return connection


def corpus_documents(benchmark: Benchmark) -> tuple[str, list[Document]]:
    """Identify the exact corpus/chunker/model combination used by the real index."""
    documents = [
        document
        for path, content in sorted(benchmark.corpus.items())
        for document in chunk_file(path, content)
    ]
    settings = get_settings()
    configuration = {
        "commit": benchmark.dataset.corpus_commit_sha,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap": CHUNK_OVERLAP,
        "files": {p: digest(c) for p, c in sorted(benchmark.corpus.items())},
        "chunks": [d.metadata for d in documents],
    }
    if settings.embeddings_provider == "ollama":
        configuration["embedding_profile"] = settings.embedding_profile
    identity = json.dumps(
        configuration,
        sort_keys=True,
    )
    return "eval_" + digest(identity), documents


async def verify_index(
    connection: asyncpg.Connection,
    benchmark: Benchmark,
    repository_id: str,
    documents: list[Document],
) -> bool:
    """Reject stale or corrupted persisted corpus/chunk metadata before scoring."""
    repository = await connection.fetchrow(
        'SELECT "indexedCommitSha", "indexStatus" FROM "Repository" WHERE "id"=$1',
        repository_id,
    )
    if not repository or repository["indexStatus"] != "READY":
        return False
    if repository["indexedCommitSha"] != benchmark.dataset.corpus_commit_sha:
        raise ValueError("Persisted corpus commit does not match the dataset.")
    files = await connection.fetch(
        'SELECT "path", "content", "contentSha" FROM "RepositoryFile" '
        'WHERE "repositoryId"=$1',
        repository_id,
    )
    actual = {str(r["path"]): str(r["content"]) for r in files}
    if actual != benchmark.corpus or any(
        digest(str(r["content"])) != r["contentSha"] for r in files
    ):
        raise ValueError("Persisted source files do not match the frozen corpus.")
    chunks = await connection.fetch(
        'SELECT "path", "startLine", "endLine", "content", "contentSha", '
        '"embeddingProfile", vector_dims("embedding") AS embedding_dimensions '
        'FROM "RepositoryChunk" WHERE "repositoryId"=$1',
        repository_id,
    )
    expected_chunks = sorted(
        (
            d.metadata["path"],
            d.metadata["start_line"],
            d.metadata["end_line"],
            d.page_content,
            d.metadata["content_sha"],
        )
        for d in documents
    )
    actual_chunks = sorted(
        (
            str(r["path"]),
            int(r["startLine"]),
            int(r["endLine"]),
            str(r["content"]),
            str(r["contentSha"]),
        )
        for r in chunks
    )
    if actual_chunks != expected_chunks:
        raise ValueError("Persisted chunks do not match the indexing configuration.")
    return not any(
        row["embeddingProfile"] != get_settings().embedding_profile
        or row["embedding_dimensions"] != get_settings().embedding_dimensions
        for row in chunks
    )


async def prepare_index(
    connection: asyncpg.Connection,
    benchmark: Benchmark,
    embeddings: Embeddings,
) -> str:
    """Idempotently seed the isolated corpus through production index writes."""
    repository_id, documents = corpus_documents(benchmark)
    if await verify_index(connection, benchmark, repository_id, documents):
        return repository_id
    vectors = await embeddings.aembed_documents([d.page_content for d in documents])
    owner, name = benchmark.dataset.repository.split("/", 1)
    await connection.execute(
        """INSERT INTO "Repository"
           ("id", "githubRepositoryId", "owner", "name", "fullName", "visibility",
            "defaultBranch", "htmlUrl", "updatedAt")
           VALUES ($1,$1,$2,$3,$4,'PUBLIC','main',$5,now())
           ON CONFLICT ("id") DO NOTHING""",
        repository_id,
        owner,
        name,
        benchmark.dataset.repository,
        f"https://github.com/{benchmark.dataset.repository}",
    )
    job_id = repository_id + "_job"
    await connection.execute(
        """INSERT INTO "RepositoryIndexJob"
           ("id", "repositoryId", "trigger", "status", "updatedAt")
           VALUES ($1,$2,'CONNECT','RUNNING',now()) ON CONFLICT ("id") DO UPDATE
           SET "status"='RUNNING', "updatedAt"=now()""",
        job_id,
        repository_id,
    )
    files = [
        {
            "path": path,
            "blob_sha": benchmark.blobs[path],
            "content_sha": digest(content),
            "byte_size": len(content.encode("utf-8")),
            "line_count": max(1, len(content.splitlines())),
            "content": content,
        }
        for path, content in sorted(benchmark.corpus.items())
    ]
    await promote_index(
        connection,
        repository_id,
        job_id,
        benchmark.dataset.corpus_commit_sha,
        files,
        documents,
        vectors,
    )
    return repository_id


async def fixture_tools(
    fixtures: Fixtures,
    scenario: str,
    records: list[ToolExecution],
) -> list[BaseTool]:
    """Use production MCP schemas with exact, default-normalized fixture matching."""
    calls = fixtures.scenarios[scenario]

    async def execute(
        name: str,
        defaults: dict[str, JsonValue],
        arguments_model: type[BaseModel],
        **arguments: JsonValue,
    ) -> dict[str, JsonValue]:
        actual = {**defaults, **arguments}
        argument_error: str | None = None
        try:
            # Use the pinned MCP registration's own Pydantic argument model.
            actual = JSON_OBJECT.validate_python(
                arguments_model.model_validate(arguments).model_dump(mode="json"),
            )
            page, per_page = actual.get("page"), actual.get("per_page")
            if not isinstance(page, int) or not isinstance(per_page, int):
                raise ValueError("Pagination values must be integers.")
            validate_page(page, "page", 1, 100)
            validate_page(per_page, "per_page", 1, 30)
            for key in ("issue_number", "pull_number"):
                if key in actual:
                    number = actual[key]
                    if not isinstance(number, int) or number < 1:
                        raise ValueError(f"{key} must be positive.")
        except ValidationError:
            argument_error = "Invalid GitHub tool argument types or enum values."
        except ValueError as error:
            argument_error = str(error)
        ignored = {"page", "per_page"}
        search_results = [
            call.result.get("items") for call in calls if call.tool == name
        ]
        if not any(
            isinstance(items, list) and len(items) > 1 for items in search_results
        ):
            ignored.update({"sort", "order"})
        selectors = {k: v for k, v in actual.items() if k not in ignored}
        if isinstance(selectors.get("query"), str):
            selectors["query"] = " ".join(str(selectors["query"]).casefold().split())
        matched = next(
            (
                call
                for call in calls
                if argument_error is None
                and call.tool == name
                and any(
                    json.dumps(selectors, sort_keys=True)
                    == json.dumps(
                        {
                            k: (
                                " ".join(v.casefold().split())
                                if k == "query" and isinstance(v, str)
                                else v
                            )
                            for k, v in {**defaults, **variant}.items()
                            if k not in ignored
                        },
                        sort_keys=True,
                    )
                    for variant in call.arguments
                )
            ),
            None,
        )
        record = ToolExecution(
            id=f"tool:{len(records) + 1}",
            name=name,
            arguments=actual,
            result=matched.result.copy() if matched else {},
            error=argument_error
            or (matched.error if matched else "Unmatched deterministic fixture call."),
            fixture_miss=matched is None and argument_error is None,
        )
        if name in {"search_issues", "search_pull_requests"} and matched:
            repository = record.result.get("repository")
            query = actual.get("query")
            if isinstance(repository, str) and isinstance(query, str):
                owner, repo = repository.split("/", 1)
                state: Literal["all", "closed", "open"] = (
                    "all"
                    if actual.get("state") == "all"
                    else "closed"
                    if actual.get("state") == "closed"
                    else "open"
                )
                record.result["query"] = search_query(
                    owner,
                    repo,
                    "issue" if name == "search_issues" else "pr",
                    query,
                    state,
                )
        if record.error is None:
            page, per_page = actual.get("page"), actual.get("per_page")
            if isinstance(page, int) and isinstance(per_page, int):
                for key in ("items", "comments", "files"):
                    items = record.result.get(key)
                    if isinstance(items, list):
                        start = (page - 1) * per_page
                        record.result[key] = items[start : start + per_page]
        records.append(record)
        if record.error:
            raise ToolException(record.error)
        return record.result

    tools: list[BaseTool] = []

    def bind(
        name: str,
        defaults: dict[str, JsonValue],
        arguments_model: type[BaseModel],
    ) -> Callable[..., Awaitable[dict[str, JsonValue]]]:
        # LangGraph inspects get_type_hints(), which rejects functools.partial.
        async def invoke(**arguments: JsonValue) -> dict[str, JsonValue]:
            return await execute(name, defaults, arguments_model, **arguments)

        return invoke

    for tool in await mcp.list_tools():
        annotations = tool.annotations.model_dump() if tool.annotations else {}
        policy = parse_tool_policy(tool.meta, annotations)
        if policy is None or policy.capability != "github.read":
            continue
        registration = mcp._tool_manager.get_tool(tool.name)
        if registration is None:
            raise ValueError("MCP tool registration is missing.")
        schema = JSON_OBJECT.validate_python(tool.inputSchema)
        defaults: dict[str, JsonValue] = {}
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, value in properties.items():
                if isinstance(value, dict) and "default" in value:
                    defaults[key] = value["default"]
        tools.append(
            StructuredTool(
                name=tool.name,
                description=tool.description or "",
                args_schema=schema,
                coroutine=bind(tool.name, defaults, registration.fn_metadata.arg_model),
                handle_tool_error=True,
                metadata={**annotations, "_meta": tool.meta},
            )
        )
    return tools


class SingleAgentRunner:
    """Adapt current retrieval/generation to the architecture-independent contract."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        embeddings: Embeddings,
        repository_id: str,
        commit_sha: str,
        repository: str,
        tools: list[BaseTool],
        tool_records: list[ToolExecution],
        *,
        retrieval_only: bool = False,
        vector_k: int = VECTOR_CANDIDATES,
        keyword_k: int = KEYWORD_CANDIDATES,
        rerank: bool = False,
    ) -> None:
        self.retriever = RepoMindParentHybridRetriever(
            connection=connection,
            embeddings=embeddings,
            repository_id=repository_id,
            vector_k=vector_k,
            keyword_k=keyword_k,
            rerank=rerank,
        )
        self.commit_sha, self.repository = commit_sha, repository
        self.tools, self.tool_records = tools, tool_records
        self.retrieval_only = retrieval_only

    async def run(self, question: str) -> EvaluationResult:
        """Execute the real retriever and shared streaming agent with empty history."""
        documents = await self.retriever.ainvoke(question)
        contexts: list[CodeContext] = []
        for index, document in enumerate(documents, start=1):
            path = document.metadata.get("path")
            first = document.metadata.get("start_line")
            last = document.metadata.get("end_line")
            if (
                not isinstance(path, str)
                or not isinstance(first, int)
                or not isinstance(
                    last,
                    int,
                )
            ):
                raise ValueError(
                    "Production retriever returned invalid source metadata."
                )
            contexts.append(
                CodeContext(
                    id=f"code:{self.commit_sha}:{path}:{first}:{last}",
                    corpus_commit_sha=self.commit_sha,
                    source_index=index,
                    path=path,
                    start_line=first,
                    end_line=last,
                    content=document.page_content,
                )
            )
        result = EvaluationResult(final_contexts=contexts, evidence_seen=contexts)
        if not self.retrieval_only:
            parts: list[str] = []
            async for event, value in stream_repository_answer(
                question,
                [],
                documents,
                self.tools,
            ):
                if event == "delta":
                    parts.append(value)
            result.answer = "".join(parts).strip()
            if not result.answer:
                raise ValueError("The production agent generated no answer text.")
        result.tool_executions = list(self.tool_records)
        result.citation_catalog = build_catalog(result, self.repository)
        return result


def score_final_evidence_coverage(
    question: Question,
    result: EvaluationResult,
    corpus: dict[str, str],
    commit_sha: str,
) -> tuple[Metric, dict[str, float]]:
    """Measure required evidence covered by all observed windows, without a @6 label."""
    observed = set().union(
        *(
            visible_lines(context, corpus, commit_sha)
            for context in result.evidence_seen
        )
    )
    overlaps = {
        unit.id: unit_overlap(unit, observed) for unit in question.required_evidence
    }
    return ratio(
        sum(value == 1 for value in overlaps.values()), len(overlaps)
    ), overlaps


class SpecializedAgentRunner:
    """Evaluate one specialist through Runner in the isolated database."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        embeddings: Embeddings,
        repository_id: str,
        commit_sha: str,
        repository: str,
        role: Literal["CODE", "GITHUB"],
        model: BaseChatModel,
        tools: list[BaseTool],
        tool_records: list[ToolExecution],
        *,
        limits: AgentLimits | None = None,
        rerank: bool = False,
    ) -> None:
        self.connection, self.embeddings = connection, embeddings
        self.repository_id, self.commit_sha, self.repository = (
            repository_id,
            commit_sha,
            repository,
        )
        self.role, self.model, self.tools = role, model, tools
        self.tool_records, self.rerank = tool_records, rerank
        self.limits = limits or AgentLimits()
        self.last_findings: AgentFindings | None = None
        self.last_result = EvaluationResult()
        self.first_query: str | None = None

    async def run(self, question: str) -> EvaluationResult:
        """Seed temporary eval access, run the specialist, then remove access."""
        connection = self.connection
        if not await connection.fetchval(
            "SELECT current_database() = 'repomind_eval' "
            "AND current_user = 'repomind_eval'"
        ):
            raise ValueError(
                "Specialist evaluation requires the dedicated eval database and role."
            )
        user_id = f"eval_agent_{uuid.uuid4().hex}"
        try:
            async with connection.transaction():
                await connection.execute(
                    'INSERT INTO "User" ("id", "clerkId", "updatedAt") '
                    "VALUES ($1,$1,now())",
                    user_id,
                )
                await connection.execute(
                    'INSERT INTO "UserRepository" '
                    '("id", "userId", "repositoryId", "updatedAt") '
                    "VALUES ($1,$2,$3,now())",
                    user_id + "_access",
                    user_id,
                    self.repository_id,
                )
            registry = ToolRegistry(
                RequestScope(
                    user_id=user_id,
                    repository_id=self.repository_id,
                    capabilities=frozenset({"code.read", "github.read"}),
                )
            )
            code_tools = CodeTools(
                registry,
                self.embeddings,
                self.commit_sha,
                connection=connection,
                rerank=self.rerank,
            )
            if self.role == "CODE":
                findings = await investigate_code(
                    self.model,
                    registry,
                    code_tools,
                    question,
                    limits=self.limits,
                )
            else:
                for tool in self.tools:
                    metadata = tool.metadata or {}
                    policy = parse_tool_policy(metadata.get("_meta"), metadata)
                    if policy is None or policy.capability != "github.read":
                        raise ValueError("GitHub fixture tool has an invalid policy.")
                    registry.register_local(tool, policy, code_tools.authorize)
                findings = await investigate_github(
                    self.model,
                    registry,
                    question,
                    limits=self.limits,
                )
            self.last_findings, self.first_query = findings, code_tools.first_query
            contexts = [
                CodeContext.model_validate(source.model_dump(exclude={"url"}))
                for source in findings.evidence
            ]
            visible_ids = {context.id for context in contexts}
            self.last_result = EvaluationResult(
                answer=findings.answer,
                final_contexts=[
                    CodeContext.model_validate(source.model_dump(exclude={"url"}))
                    for source in code_tools.first_pass
                    if source.id in visible_ids
                ],
                evidence_seen=contexts,
                tool_executions=(
                    list(self.tool_records)
                    if self.role == "GITHUB"
                    else [
                        ToolExecution(
                            id=record.id,
                            name=record.name,
                            arguments=record.arguments,
                            result=record.result,
                            error=record.message.text
                            if record.message.status == "error"
                            else None,
                        )
                        for record in findings.tool_executions
                    ]
                ),
            )
            self.last_result.citation_catalog = build_catalog(
                self.last_result, self.repository
            )
            if self.role == "CODE":
                self.last_result.citation_catalog.update(findings.citation_catalog)
            if findings.status == "failed":
                raise RuntimeError(
                    "Specialized investigation failed; inspect its structured findings."
                )
            return self.last_result
        finally:
            # All agent/tool scopes have unwound before eval-only writes resume.
            # UserRepository cascades on delete, including cancellation paths.
            with anyio.CancelScope(shield=True):
                await connection.execute('DELETE FROM "User" WHERE "id"=$1', user_id)
