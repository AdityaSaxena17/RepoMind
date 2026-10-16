import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from app import worker
from app.config import get_settings


@pytest.mark.parametrize("indexed_sha", [None, "a" * 40, "b" * 40])
def test_worker_only_rebuilds_when_live_commit_differs(
    monkeypatch: pytest.MonkeyPatch,
    indexed_sha: str | None,
) -> None:
    live_sha = "a" * 40
    calls: list[str] = []
    writes: list[str] = []

    class Connection:
        async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
            return {
                "owner": "example",
                "name": "project",
                "defaultBranch": "main",
                "indexStatus": "UPDATING",
                "indexedCommitSha": indexed_sha,
            }

        async def execute(self, sql: str, *args: Any) -> None:
            writes.append(sql)

        @asynccontextmanager
        async def transaction(self) -> Any:
            yield

        async def close(self) -> None:
            calls.append("close")

    async def connect(url: str) -> Connection:
        return Connection()

    async def head(*args: Any) -> str:
        calls.append("head")
        return live_sha

    async def tree(*args: Any) -> list[dict[str, Any]]:
        calls.append("tree")
        return [{"type": "blob", "path": "app.py", "sha": "c" * 40, "size": 10}]

    async def content(*args: Any) -> str:
        calls.append("content")
        assert args[-1] == live_sha
        return "print('ok')"

    class Embeddings:
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append("embed")
            return [[1.0] for _ in texts]

    async def promote(*args: Any) -> None:
        calls.append("promote")
        assert args[3] == live_sha and len(args[4]) == len(args[5]) == len(args[6]) == 1

    monkeypatch.setattr(worker.asyncpg, "connect", connect)
    monkeypatch.setattr(worker, "fetch_default_branch_head", head)
    monkeypatch.setattr(worker, "fetch_tree", tree)
    monkeypatch.setattr(worker, "fetch_file_content", content)
    monkeypatch.setattr(worker, "create_embedding_model", Embeddings)
    monkeypatch.setattr(worker, "promote_index", promote)
    result = asyncio.run(
        worker.index_repository({}, {"repository_id": "repo", "job_id": "job"})
    )
    assert result["commit_sha"] == live_sha
    if indexed_sha == live_sha:
        assert result["status"] == "unchanged"
        assert calls == ["head", "close"]
        assert not any('"lastIndexedAt"' in sql or "DELETE" in sql for sql in writes)
    else:
        assert result["status"] == "indexed"
        assert calls == ["head", "tree", "content", "embed", "promote", "close"]


def test_worker_github_headers_use_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_real")
    get_settings.cache_clear()

    assert worker.github_headers()["Authorization"] == "Bearer github_pat_real"

    get_settings.cache_clear()


def test_worker_github_headers_omit_placeholder_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_optional_public_read_token")
    get_settings.cache_clear()

    assert "Authorization" not in worker.github_headers()

    get_settings.cache_clear()
