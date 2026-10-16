import asyncio
from typing import Any

import pytest

from app import github_mcp_server


def run_async(value: Any) -> Any:
  return asyncio.run(value)


def test_mcp_lists_exact_github_tools() -> None:
  tools = run_async(github_mcp_server.mcp.list_tools())

  assert {tool.name for tool in tools} == {
    "search_issues",
    "issue_read",
    "search_pull_requests",
    "pull_request_read",
    "get_repository_head",
  }

  for tool in tools:
    properties = tool.inputSchema.get("properties", {})
    assert "owner" not in properties
    assert "repo" not in properties


@pytest.mark.parametrize("denied", [False, True])
def test_live_head_lookup_is_authorized_and_scoped(
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    calls = []
    sha = "b" * 40

    async def scope() -> dict[str, str]:
        calls.append("authorize")
        if denied:
            raise github_mcp_server.AuthorizationError("Denied.")
        return {
            "owner": "octo",
            "repo": "hello",
            "html_url": "https://github.com/octo/hello",
        }

    async def get(path: str) -> Any:
        calls.append(path)
        return [{"sha": sha, "html_url": f"https://github.com/octo/hello/commit/{sha}"}]

    monkeypatch.setattr(github_mcp_server, "repository_scope", scope)
    monkeypatch.setattr(github_mcp_server, "github_json", get)
    if denied:
        with pytest.raises(github_mcp_server.AuthorizationError):
            run_async(github_mcp_server.get_repository_head())
        assert calls == ["authorize"]
    else:
        result = run_async(github_mcp_server.get_repository_head())
        assert result["commit_sha"] == sha
        assert result["url"] == f"https://github.com/octo/hello/commit/{sha}"
        assert calls == ["authorize", "/repos/octo/hello/commits?per_page=1"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{}],
        [
            {
                "sha": "b" * 40,
                "html_url": "https://github.com/other/repo/commit/" + "b" * 40,
            }
        ],
    ],
)
def test_live_head_rejects_missing_or_cross_repository_metadata(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
) -> None:
    async def scope() -> dict[str, str]:
        return {
            "owner": "octo",
            "repo": "hello",
            "html_url": "https://github.com/octo/hello",
        }

    async def get(path: str) -> Any:
        return payload

    monkeypatch.setattr(github_mcp_server, "repository_scope", scope)
    monkeypatch.setattr(github_mcp_server, "github_json", get)
    with pytest.raises(ValueError):
        run_async(github_mcp_server.get_repository_head())


def test_repository_scope_requires_locked_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPOMIND_MCP_USER_ID", raising=False)
    monkeypatch.delenv("REPOMIND_MCP_REPOSITORY_ID", raising=False)

    with pytest.raises(RuntimeError, match="scope is missing"):
        run_async(github_mcp_server.repository_scope())


def test_repository_scope_validates_user_access(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class FakeConnection:
    async def fetchrow(self, _query: str, repository_id: str, user_id: str) -> None:
      assert repository_id == "repo_123"
      assert user_id == "user_123"
      return None

    async def close(self) -> None:
      return None

  async def fake_connect(_database_url: str) -> FakeConnection:
    return FakeConnection()

  monkeypatch.setenv("REPOMIND_MCP_USER_ID", "user_123")
  monkeypatch.setenv("REPOMIND_MCP_REPOSITORY_ID", "repo_123")
  monkeypatch.setattr(github_mcp_server.asyncpg, "connect", fake_connect)

  with pytest.raises(RuntimeError, match="no longer accessible"):
    run_async(github_mcp_server.repository_scope())


def test_search_issues_scopes_to_repo_and_excludes_pull_requests(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  async def fake_scope() -> dict[str, str]:
    return {
      "owner": "octo",
      "repo": "hello",
      "html_url": "https://github.com/octo/hello",
    }

  async def fake_github_json(path: str) -> dict[str, Any]:
    calls.append(path)
    return {
      "total_count": 2,
      "items": [
        {
          "number": 1,
          "title": "Bug",
          "html_url": "https://github.com/octo/hello/issues/1",
          "state": "open",
          "user": {"login": "alice"},
          "labels": [{"name": "bug"}],
        },
        {
          "number": 2,
          "title": "PR",
          "html_url": "https://github.com/octo/hello/pull/2",
          "state": "open",
          "pull_request": {},
        },
      ],
    }

  monkeypatch.setattr(github_mcp_server, "repository_scope", fake_scope)
  monkeypatch.setattr(github_mcp_server, "github_json", fake_github_json)

  result = run_async(
    github_mcp_server.search_issues(query="crash", state="all"),
  )

  assert "repo%3Aocto%2Fhello+is%3Aissue+crash" in calls[0]
  assert result["repository"] == "octo/hello"
  assert [item["number"] for item in result["items"]] == [1]


def test_search_pull_requests_scopes_to_repo_and_excludes_issues(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  async def fake_scope() -> dict[str, str]:
    return {
      "owner": "octo",
      "repo": "hello",
      "html_url": "https://github.com/octo/hello",
    }

  async def fake_github_json(path: str) -> dict[str, Any]:
    calls.append(path)
    return {
      "total_count": 2,
      "items": [
        {
          "number": 1,
          "title": "Issue",
          "html_url": "https://github.com/octo/hello/issues/1",
          "state": "open",
        },
        {
          "number": 2,
          "title": "PR",
          "html_url": "https://github.com/octo/hello/pull/2",
          "state": "open",
          "pull_request": {},
        },
      ],
    }

  monkeypatch.setattr(github_mcp_server, "repository_scope", fake_scope)
  monkeypatch.setattr(github_mcp_server, "github_json", fake_github_json)

  result = run_async(
    github_mcp_server.search_pull_requests(query="feature", state="all"),
  )

  assert "repo%3Aocto%2Fhello+is%3Apr+feature" in calls[0]
  assert result["repository"] == "octo/hello"
  assert [item["number"] for item in result["items"]] == [2]


def test_issue_read_rejects_pull_request_payload(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def fake_scope() -> dict[str, str]:
    return {
      "owner": "octo",
      "repo": "hello",
      "html_url": "https://github.com/octo/hello",
    }

  async def fake_github_json(_path: str) -> dict[str, Any]:
    return {
      "number": 2,
      "title": "PR",
      "html_url": "https://github.com/octo/hello/pull/2",
      "state": "open",
      "pull_request": {},
    }

  monkeypatch.setattr(github_mcp_server, "repository_scope", fake_scope)
  monkeypatch.setattr(github_mcp_server, "github_json", fake_github_json)

  with pytest.raises(RuntimeError, match="issue was not found"):
    run_async(github_mcp_server.issue_read(issue_number=2))
