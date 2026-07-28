import asyncio
import base64
import hashlib
import time
import uuid
from typing import Any

import asyncpg
import httpx
import jwt
import tiktoken

from .config import get_settings

MAX_FILE_BYTES = 1_048_576
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
EMBEDDING_DIMENSIONS = 1536

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


def create_github_jwt() -> str:
    settings = get_settings()

    if not settings.github_app_id or not settings.github_app_private_key:
        raise RuntimeError("GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY are required.")

    now = int(time.time())
    return jwt.encode(
        {
            "iat": now - 60,
            "exp": now + 540,
            "iss": settings.github_app_id,
        },
        settings.github_app_private_key.replace("\\n", "\n"),
        algorithm="RS256",
    )


async def github_json(
    client: httpx.AsyncClient,
    path: str,
    token: str,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    response = await client.request(
        method,
        f"{settings.github_api_base_url}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "RepoMind",
            "X-GitHub-Api-Version": settings.github_api_version,
        },
        json=json_body,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected payload.")

    return payload


async def create_installation_token(
    client: httpx.AsyncClient,
    installation_id: str,
    repository_name: str,
) -> str:
    payload = await github_json(
        client,
        f"/app/installations/{installation_id}/access_tokens",
        create_github_jwt(),
        method="POST",
        json_body={"repositories": [repository_name]},
    )
    token = payload.get("token")

    if not isinstance(token, str):
        raise RuntimeError("GitHub did not return an installation token.")

    return token


async def fetch_default_branch_head(
    client: httpx.AsyncClient,
    owner: str,
    name: str,
    branch: str,
    token: str,
) -> str:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/commits/{branch}",
        token,
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
    token: str,
) -> list[dict[str, Any]]:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/git/trees/{commit_sha}?recursive=1",
        token,
    )
    tree = payload.get("tree")

    if not isinstance(tree, list):
        raise RuntimeError("GitHub tree response was incomplete.")

    if payload.get("truncated") is True:
        root = await github_json(
            client,
            f"/repos/{owner}/{name}/git/trees/{commit_sha}",
            token,
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
                token,
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
    token: str,
) -> str | None:
    payload = await github_json(
        client,
        f"/repos/{owner}/{name}/contents/{path}?ref={commit_sha}",
        token,
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


def chunk_file(path: str, content: str) -> list[dict[str, Any]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    lines = content.splitlines()
    line_offsets: list[int] = []
    offset = 0

    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1

    tokens = encoding.encode(content)
    chunks: list[dict[str, Any]] = []
    start_token = 0

    while start_token < len(tokens):
        end_token = min(start_token + CHUNK_TOKENS, len(tokens))
        text = encoding.decode(tokens[start_token:end_token])
        start_char = len(encoding.decode(tokens[:start_token]))
        end_char = start_char + len(text)
        start_line = 1
        end_line = max(1, len(lines))

        for index, line_offset in enumerate(line_offsets):
            if line_offset <= start_char:
                start_line = index + 1
            if line_offset <= end_char:
                end_line = index + 1

        chunks.append(
            {
                "path": path,
                "content": text,
                "content_sha": sha256_text(text),
                "start_line": start_line,
                "end_line": end_line,
            },
        )

        if end_token == len(tokens):
            break

        start_token = max(end_token - CHUNK_OVERLAP, start_token + 1)

    return chunks


async def embed_chunks(chunks: list[dict[str, Any]]) -> list[list[float]]:
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for repository indexing.")

    embeddings: list[list[float]] = []

    async with httpx.AsyncClient() as client:
        for index in range(0, len(chunks), 64):
            batch = chunks[index : index + 64]
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openai_embeddings_model,
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "input": [chunk["content"] for chunk in batch],
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None

            if not isinstance(data, list):
                raise RuntimeError("OpenAI embeddings response was incomplete.")

            for item in data:
                if not isinstance(item, dict) or not isinstance(
                    item.get("embedding"),
                    list,
                ):
                    raise RuntimeError("OpenAI embedding item was incomplete.")
                embeddings.append([float(value) for value in item["embedding"]])

    return embeddings


async def promote_index(
    connection: asyncpg.Connection,
    repository_id: str,
    job_id: str,
    commit_sha: str,
    files: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
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

        for index, chunk in enumerate(chunks):
            file_id = file_ids[str(chunk["path"])]
            vector = "[" + ",".join(str(value) for value in embeddings[index]) + "]"
            await connection.execute(
                """
                INSERT INTO "RepositoryChunk"
                  ("id", "repositoryId", "fileId", "chunkIndex", "path",
                   "startLine", "endLine", "content", "contentSha", "embedding")
                VALUES
                  ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
                """,
                uuid.uuid4().hex,
                repository_id,
                file_id,
                index,
                chunk["path"],
                chunk["start_line"],
                chunk["end_line"],
                chunk["content"],
                chunk["content_sha"],
                vector,
            )

        await connection.execute(
            """
            UPDATE "Repository"
            SET "indexStatus" = 'READY',
                "indexProgress" = 100,
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
                "progress" = 100,
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
    await asyncio.sleep(0)
    repository_id = str(payload["repository_id"])
    job_id = str(payload["job_id"])
    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)

    try:
        repository = await connection.fetchrow(
            """
            SELECT "owner", "name", "defaultBranch", "githubInstallationId",
                   "indexStatus"
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
            token = await create_installation_token(
                client,
                str(repository["githubInstallationId"]),
                str(repository["name"]),
            )
            commit_sha = await fetch_default_branch_head(
                client,
                str(repository["owner"]),
                str(repository["name"]),
                str(repository["defaultBranch"]),
                token,
            )
            tree = await fetch_tree(
                client,
                str(repository["owner"]),
                str(repository["name"]),
                commit_sha,
                token,
            )
            files: list[dict[str, Any]] = []
            chunks: list[dict[str, Any]] = []

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
                    token,
                )

                if content is None or "-----BEGIN " in content[:2000]:
                    continue

                files.append(
                    {
                        "path": path,
                        "blob_sha": blob_sha,
                        "content_sha": sha256_text(content),
                        "byte_size": len(content.encode("utf-8")),
                        "line_count": max(1, len(content.splitlines())),
                        "content": content,
                    },
                )
                chunks.extend(chunk_file(path, content))

        embeddings = await embed_chunks(chunks) if chunks else []
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
