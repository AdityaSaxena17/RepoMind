import base64
import hashlib
import uuid
from typing import Any

import asyncpg
import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings
from .retrieval import RepoMindRepositoryVectorStore, create_embedding_model

MAX_FILE_BYTES = 1_048_576
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

SKIPPED_DIR_PARTS = {
    ".git",
    ".next",
    ".turbo",
    ".venv",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
SKIPPED_SUFFIXES = {
    ".env",
    ".lock",
    ".map",
    ".min.css",
    ".min.js",
    ".pem",
    ".pfx",
    ".p12",
}
SKIPPED_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "cargo.lock",
    "gemfile.lock",
}


def redacted_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:300] if message else error.__class__.__name__


def should_skip_path(path: str, size: int | None) -> bool:
    normalized = path.lower()
    parts = set(normalized.split("/"))

    if size is not None and size > MAX_FILE_BYTES:
        return True

    if parts.intersection(SKIPPED_DIR_PARTS):
        return True

    if normalized in SKIPPED_FILENAMES:
        return True

    if any(normalized.endswith(suffix) for suffix in SKIPPED_SUFFIXES):
        return True

    return "private key" in normalized or normalized.startswith(".env")


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def github_headers() -> dict[str, str]:
    """Build GitHub headers for repository indexing with optional auth."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "RepoMind",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = get_settings().github_token

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


async def github_json(
    client: httpx.AsyncClient,
    path: str,
) -> dict[str, Any]:
    response = await client.request(
        "GET",
        f"{GITHUB_API_BASE_URL}{path}",
        headers=github_headers(),
        timeout=30,
    )

    if response.status_code == 404:
        raise RuntimeError("GitHub repository was not found or is not public.")

    if response.status_code == 403:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_at = response.headers.get("X-RateLimit-Reset")
        auth_state = (
            "authenticated" if get_settings().github_token else "unauthenticated"
        )
        detail = (
            f"GitHub rate limit blocked this {auth_state} request"
            f" (remaining={remaining or 'unknown'}, reset={reset_at or 'unknown'})."
        )
        raise RuntimeError(detail)

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected payload.")

    return payload


async def fetch_default_branch_head(
    client: httpx.AsyncClient,
    owner: str,
    name: str,
    branch: str,
) -> str:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/commits/{branch}",
    )
    sha = payload.get("sha")

    if not isinstance(sha, str):
        raise RuntimeError("GitHub commit response did not include a sha.")

    return sha


async def fetch_tree(
    client: httpx.AsyncClient,
    owner: str,
    name: str,
    commit_sha: str,
) -> list[dict[str, Any]]:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/git/trees/{commit_sha}?recursive=1",
    )
    tree = payload.get("tree")

    if not isinstance(tree, list):
        raise RuntimeError("GitHub tree response was incomplete.")

    if payload.get("truncated") is True:
        root = await github_json(
            client,
            f"/repos/{owner}/{name}/git/trees/{commit_sha}",
        )
        root_tree = root.get("tree")

        if not isinstance(root_tree, list):
            raise RuntimeError("GitHub non-recursive tree response was incomplete.")

        complete: list[dict[str, Any]] = []
        pending = [entry for entry in root_tree if isinstance(entry, dict)]

        while pending:
            entry = pending.pop()
            entry_type = entry.get("type")

            if entry_type == "blob":
                complete.append(entry)
                continue

            if entry_type != "tree" or not isinstance(entry.get("sha"), str):
                continue

            subtree = await github_json(
                client,
                f"/repos/{owner}/{name}/git/trees/{entry['sha']}",
            )
            subtree_items = subtree.get("tree")

            if not isinstance(subtree_items, list):
                raise RuntimeError("GitHub subtree response was incomplete.")

            prefix = entry.get("path") if isinstance(entry.get("path"), str) else ""

            for subtree_item in subtree_items:
                if not isinstance(subtree_item, dict):
                    continue

                item_path = subtree_item.get("path")

                if isinstance(item_path, str) and prefix:
                    subtree_item["path"] = f"{prefix}/{item_path}"

                pending.append(subtree_item)

        return complete

    return [entry for entry in tree if isinstance(entry, dict)]


async def fetch_file_content(
    client: httpx.AsyncClient,
    owner: str,
    name: str,
    path: str,
    commit_sha: str,
) -> str | None:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/contents/{path}?ref={commit_sha}",
    )

    if payload.get("encoding") != "base64" or not isinstance(
        payload.get("content"),
        str,
    ):
        return None

    try:
        raw = base64.b64decode(str(payload["content"]), validate=False)
        return raw.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


def chunk_file(path: str, content: str) -> list[Document]:
    """Split source text and record inclusive line ranges for parent expansion."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    lines = content.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0

    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    chunks: list[Document] = []
    documents = splitter.create_documents([content])

    for document in documents:
        text = document.page_content
        start_index = document.metadata.get("start_index")
        start_char = start_index if isinstance(start_index, int) else content.find(text)
        # The token splitter's offset search subtracts token overlap as characters
        # and can return -1. Locate its unchanged text before expanding parent lines.
        if start_char < 0:
            start_char = content.find(text)
        if start_char < 0:
            raise ValueError(f"Chunk text does not occur in source: {path}.")
        end_char = start_char + len(text) - 1
        start_line = 1
        end_line = max(1, len(lines))

        for index, line_offset in enumerate(line_offsets):
            if line_offset <= start_char:
                start_line = index + 1
            if line_offset <= end_char:
                end_line = index + 1

        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "path": path,
                    "content_sha": sha256_text(text),
                    "start_line": start_line,
                    "end_line": end_line,
                },
            ),
        )

    return chunks


async def promote_index(
    connection: asyncpg.Connection,
    repository_id: str,
    job_id: str,
    commit_sha: str,
    files: list[dict[str, Any]],
    chunks: list[Document],
    embeddings: list[list[float]],
) -> None:
    async with connection.transaction():
        await connection.execute(
            'DELETE FROM "RepositoryFile" WHERE "repositoryId" = $1',
            repository_id,
        )

        file_ids: dict[str, str] = {}

        for file in files:
            row = await connection.fetchrow(
                """
                INSERT INTO "RepositoryFile"
                  ("id", "repositoryId", "path", "blobSha", "contentSha",
                   "byteSize", "lineCount", "content", "updatedAt")
                VALUES
                  ($1, $2, $3, $4, $5, $6, $7, $8, now())
                RETURNING "id"
                """,
                uuid.uuid4().hex,
                repository_id,
                file["path"],
                file["blob_sha"],
                file["content_sha"],
                file["byte_size"],
                file["line_count"],
                file["content"],
            )
            file_ids[str(file["path"])] = str(row["id"])

        vector_store = RepoMindRepositoryVectorStore(
            connection=connection,
            embeddings=create_embedding_model(),
            repository_id=repository_id,
            file_ids_by_path=file_ids,
        )
        await vector_store.awrite_embedded_documents(chunks, embeddings)

        await connection.execute(
            """
            UPDATE "Repository"
            SET "indexStatus" = 'READY',
                "indexedCommitSha" = $2,
                "targetCommitSha" = $2,
                "indexedFileCount" = $3,
                "indexedChunkCount" = $4,
                "lastIndexedAt" = now(),
                "lastIndexError" = NULL,
                "updatedAt" = now()
            WHERE "id" = $1
            """,
            repository_id,
            commit_sha,
            len(files),
            len(chunks),
        )
        await connection.execute(
            """
            UPDATE "RepositoryIndexJob"
            SET "status" = 'SUCCEEDED',
                "finishedAt" = now(),
                "updatedAt" = now()
            WHERE "id" = $1
            """,
            job_id,
        )


async def index_repository(
    ctx: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, str]:
    repository_id = str(payload["repository_id"])
    job_id = str(payload["job_id"])
    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)

    try:
        repository = await connection.fetchrow(
            """
            SELECT "owner", "name", "defaultBranch", "indexStatus", "indexedCommitSha"
            FROM "Repository"
            WHERE "id" = $1 AND "accessStatus" = 'ACTIVE'
            """,
            repository_id,
        )

        if not repository:
            raise RuntimeError("Repository is not active or no longer exists.")

        await connection.execute(
            """
            UPDATE "RepositoryIndexJob"
            SET "status" = 'RUNNING', "startedAt" = now(), "updatedAt" = now()
            WHERE "id" = $1
            """,
            job_id,
        )

        async with httpx.AsyncClient() as client:
            commit_sha = await fetch_default_branch_head(
                client,
                str(repository["owner"]),
                str(repository["name"]),
                str(repository["defaultBranch"]),
            )
            if repository["indexedCommitSha"] == commit_sha:
                # HEAD can change between the web check and dequeuing. Do not
                # replace files, embeddings, timestamps or cache revisions here.
                async with connection.transaction():
                    await connection.execute(
                        'UPDATE "Repository" SET "indexStatus"=\'READY\', '
                        '"targetCommitSha"=$2, "lastIndexError"=NULL, '
                        '"updatedAt"=now() '
                        'WHERE "id"=$1',
                        repository_id,
                        commit_sha,
                    )
                    await connection.execute(
                        'UPDATE "RepositoryIndexJob" SET "status"=\'SUCCEEDED\', '
                        '"finishedAt"=now(), "updatedAt"=now() WHERE "id"=$1',
                        job_id,
                    )
                return {
                    "repository_id": repository_id,
                    "status": "unchanged",
                    "commit_sha": commit_sha,
                }
            tree = await fetch_tree(
                client,
                str(repository["owner"]),
                str(repository["name"]),
                commit_sha,
            )
            files: list[dict[str, Any]] = []
            chunks: list[Document] = []

            for entry in tree:
                if entry.get("type") != "blob":
                    continue

                path = entry.get("path")
                blob_sha = entry.get("sha")
                size = entry.get("size")

                if not isinstance(path, str) or not isinstance(blob_sha, str):
                    continue

                file_size = int(size) if isinstance(size, int) else None

                if should_skip_path(path, file_size):
                    continue

                content = await fetch_file_content(
                    client,
                    str(repository["owner"]),
                    str(repository["name"]),
                    path,
                    commit_sha,
                )

                if content is None or "-----BEGIN " in content[:2000]:
                    continue

                file_record = {
                    "path": path,
                    "blob_sha": blob_sha,
                    "content_sha": sha256_text(content),
                    "byte_size": len(content.encode("utf-8")),
                    "line_count": max(1, len(content.splitlines())),
                    "content": content,
                }
                files.append(file_record)
                chunks.extend(chunk_file(path, content))

        embeddings = (
            await RepoMindRepositoryVectorStore(
                connection=connection,
                embeddings=create_embedding_model(),
                repository_id=repository_id,
            ).aembed_documents(chunks)
            if chunks
            else []
        )
        await promote_index(
            connection,
            repository_id,
            job_id,
            commit_sha,
            files,
            chunks,
            embeddings,
        )

        return {
            "repository_id": repository_id,
            "status": "indexed",
            "commit_sha": commit_sha,
        }
    except Exception as error:
        message = redacted_error(error)
        await connection.execute(
            """
            UPDATE "Repository"
            SET "indexStatus" = 'FAILED',
                "lastIndexError" = $2,
                "updatedAt" = now()
            WHERE "id" = $1
            """,
            repository_id,
            message,
        )
        await connection.execute(
            """
            UPDATE "RepositoryIndexJob"
            SET "status" = 'FAILED',
                "errorMessage" = $2,
                "finishedAt" = now(),
                "updatedAt" = now()
            WHERE "id" = $1
            """,
            job_id,
            message,
        )
        raise
    finally:
        await connection.close()


class WorkerSettings:
    functions = [index_repository]
    redis_settings = get_settings().redis_settings
    queue_name = get_settings().arq_queue_name
