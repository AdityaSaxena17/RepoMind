from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import urlencode

import asyncpg
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .config import get_settings

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_TOOL_META = {
    "repomind": {"capability": "github.read", "repository_scoped": True}
}
GITHUB_TOOL_ANNOTATIONS = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

mcp = FastMCP(
    "RepoMind GitHub",
    instructions=(
        "Read public GitHub issues and pull requests for exactly one connected "
        "RepoMind repository. Repository scope is enforced by the server."
    ),
)


def read_github_token() -> str | None:
    """Read an optional GitHub token while ignoring placeholder-like values."""
    return get_settings().github_token


def read_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def read_number(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def read_user(value: object) -> str | None:
    if not isinstance(value, dict):
        return None

    return read_string(value.get("login"))


def compact_text(value: object, limit: int = 1200) -> str | None:
    text = read_string(value)

    if text is None:
        return None

    normalized = " ".join(text.split())
    return normalized[:limit]


def validate_page(value: int, name: str, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    return value


def github_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "RepoMind",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = read_github_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


async def github_json(path: str) -> Any:
    """Fetch JSON from public GitHub REST with redacted user-facing errors."""
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{GITHUB_API_BASE_URL}{path}",
            headers=github_headers(),
        )

    if response.status_code == 401:
        raise RuntimeError("GitHub rejected the configured token.")

    if response.status_code == 403:
        raise RuntimeError("GitHub rate limit blocked this request.")

    if response.status_code == 404:
        raise RuntimeError("GitHub resource was not found.")

    if response.status_code >= 400:
        raise RuntimeError(f"GitHub request failed with status {response.status_code}.")

    return response.json()


async def github_text(path: str, accept: str) -> str:
    """Fetch text from public GitHub REST with redacted user-facing errors."""
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{GITHUB_API_BASE_URL}{path}",
            headers=github_headers(accept),
        )

    if response.status_code == 401:
        raise RuntimeError("GitHub rejected the configured token.")

    if response.status_code == 403:
        raise RuntimeError("GitHub rate limit blocked this request.")

    if response.status_code == 404:
        raise RuntimeError("GitHub resource was not found.")

    if response.status_code >= 400:
        raise RuntimeError(f"GitHub request failed with status {response.status_code}.")

    return response.text[:12000]


async def repository_scope() -> dict[str, str]:
    """Resolve the MCP repository scope and enforce RepoMind user access."""
    user_id = read_string(os.getenv("REPOMIND_MCP_USER_ID"))
    repository_id = read_string(os.getenv("REPOMIND_MCP_REPOSITORY_ID"))

    if user_id is None or repository_id is None:
        raise RuntimeError("MCP repository scope is missing.")

    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)

    try:
        row = await connection.fetchrow(
            """
            SELECT r."id", r."owner", r."name", r."htmlUrl"
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
    finally:
        await connection.close()

    if not row:
        raise RuntimeError("Repository is not active or no longer accessible.")

    return {
        "owner": str(row["owner"]),
        "repo": str(row["name"]),
        "html_url": str(row["htmlUrl"] or ""),
    }


def serialize_issue(item: dict[str, Any]) -> dict[str, Any] | None:
    number = read_number(item.get("number"))
    title = read_string(item.get("title"))
    url = read_string(item.get("html_url"))
    state = read_string(item.get("state"))

    if number is None or title is None or url is None or state is None:
        return None

    return {
        "number": number,
        "title": title,
        "state": state,
        "url": url,
        "author": read_user(item.get("user")),
        "labels": [
            label["name"]
            for label in item.get("labels", [])
            if isinstance(label, dict) and isinstance(label.get("name"), str)
        ][:10],
        "created_at": read_string(item.get("created_at")),
        "updated_at": read_string(item.get("updated_at")),
        "body": compact_text(item.get("body")),
    }


def serialize_pull_request(item: dict[str, Any]) -> dict[str, Any] | None:
    number = read_number(item.get("number"))
    title = read_string(item.get("title"))
    url = read_string(item.get("html_url"))
    state = read_string(item.get("state"))

    if number is None or title is None or url is None or state is None:
        return None

    return {
        "number": number,
        "title": title,
        "state": state,
        "url": url,
        "author": read_user(item.get("user")),
        "created_at": read_string(item.get("created_at")),
        "updated_at": read_string(item.get("updated_at")),
        "merged": item.get("merged") if isinstance(item.get("merged"), bool) else None,
        "base": item.get("base", {}).get("ref")
        if isinstance(item.get("base"), dict)
        else None,
        "head": item.get("head", {}).get("ref")
        if isinstance(item.get("head"), dict)
        else None,
        "body": compact_text(item.get("body")),
    }


def search_query(
    owner: str,
    repo: str,
    kind: Literal["issue", "pr"],
    query: str,
    state: Literal["open", "closed", "all"],
) -> str:
    parts = [f"repo:{owner}/{repo}", f"is:{kind}"]

    if state != "all":
        parts.append(f"is:{state}")

    trimmed = query.strip()
    if trimmed:
        parts.append(trimmed)

    return " ".join(parts)


def query_string(values: dict[str, object]) -> str:
    return urlencode({key: str(value) for key, value in values.items()})


@mcp.tool(meta=GITHUB_TOOL_META, annotations=GITHUB_TOOL_ANNOTATIONS)
async def search_issues(
    query: str = "",
    state: Literal["open", "closed", "all"] = "open",
    page: int = 1,
    per_page: int = 10,
    sort: Literal["comments", "created", "updated"] = "updated",
    order: Literal["asc", "desc"] = "desc",
) -> dict[str, Any]:
    """Search issues in the scoped repository."""
    scope = await repository_scope()
    page = validate_page(page, "page", 1, 100)
    per_page = validate_page(per_page, "per_page", 1, 30)
    q = search_query(scope["owner"], scope["repo"], "issue", query, state)
    params = query_string(
        {"q": q, "page": page, "per_page": per_page, "sort": sort, "order": order},
    )
    payload = await github_json(
        f"/search/issues?{params}",
    )

    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected issue search response.")

    items = [
        serialized
        for item in payload.get("items", [])
        if isinstance(item, dict) and "pull_request" not in item
        if (serialized := serialize_issue(item)) is not None
    ]

    return {
        "repository": f"{scope['owner']}/{scope['repo']}",
        "query": q,
        "total_count": read_number(payload.get("total_count")) or len(items),
        "items": items,
    }


@mcp.tool(meta=GITHUB_TOOL_META, annotations=GITHUB_TOOL_ANNOTATIONS)
async def issue_read(
    issue_number: int,
    method: Literal["get", "get_comments"] = "get",
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """Read one issue or its comments from the scoped repository."""
    scope = await repository_scope()
    page = validate_page(page, "page", 1, 100)
    per_page = validate_page(per_page, "per_page", 1, 30)

    if issue_number < 1:
        raise ValueError("issue_number must be positive.")

    if method == "get":
        payload = await github_json(
            f"/repos/{scope['owner']}/{scope['repo']}/issues/{issue_number}",
        )

        if not isinstance(payload, dict) or "pull_request" in payload:
            raise RuntimeError("GitHub issue was not found.")

        issue = serialize_issue(payload)
        if issue is None:
            raise RuntimeError("GitHub returned an incomplete issue.")

        return {"repository": f"{scope['owner']}/{scope['repo']}", "issue": issue}

    payload = await github_json(
        f"/repos/{scope['owner']}/{scope['repo']}/issues/{issue_number}/comments"
        f"?page={page}&per_page={per_page}",
    )

    if not isinstance(payload, list):
        raise RuntimeError("GitHub returned an unexpected issue comments response.")

    return {
        "repository": f"{scope['owner']}/{scope['repo']}",
        "issue_number": issue_number,
        "comments": [
            {
                "id": read_number(item.get("id")),
                "url": read_string(item.get("html_url")),
                "author": read_user(item.get("user")),
                "created_at": read_string(item.get("created_at")),
                "updated_at": read_string(item.get("updated_at")),
                "body": compact_text(item.get("body")),
            }
            for item in payload
            if isinstance(item, dict)
        ],
    }


@mcp.tool(meta=GITHUB_TOOL_META, annotations=GITHUB_TOOL_ANNOTATIONS)
async def search_pull_requests(
    query: str = "",
    state: Literal["open", "closed", "all"] = "open",
    page: int = 1,
    per_page: int = 10,
    sort: Literal["comments", "created", "updated"] = "updated",
    order: Literal["asc", "desc"] = "desc",
) -> dict[str, Any]:
    """Search pull requests in the scoped repository."""
    scope = await repository_scope()
    page = validate_page(page, "page", 1, 100)
    per_page = validate_page(per_page, "per_page", 1, 30)
    q = search_query(scope["owner"], scope["repo"], "pr", query, state)
    params = query_string(
        {"q": q, "page": page, "per_page": per_page, "sort": sort, "order": order},
    )
    payload = await github_json(
        f"/search/issues?{params}",
    )

    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected PR search response.")

    items = [
        serialized
        for item in payload.get("items", [])
        if isinstance(item, dict) and "pull_request" in item
        if (serialized := serialize_pull_request(item)) is not None
    ]

    return {
        "repository": f"{scope['owner']}/{scope['repo']}",
        "query": q,
        "total_count": read_number(payload.get("total_count")) or len(items),
        "items": items,
    }


@mcp.tool(meta=GITHUB_TOOL_META, annotations=GITHUB_TOOL_ANNOTATIONS)
async def pull_request_read(
    pull_number: int,
    method: Literal["get", "get_files", "get_diff", "get_comments"] = "get",
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """Read one pull request, files, diff, or comments from the scoped repository."""
    scope = await repository_scope()
    page = validate_page(page, "page", 1, 100)
    per_page = validate_page(per_page, "per_page", 1, 30)

    if pull_number < 1:
        raise ValueError("pull_number must be positive.")

    if method == "get":
        payload = await github_json(
            f"/repos/{scope['owner']}/{scope['repo']}/pulls/{pull_number}",
        )

        if not isinstance(payload, dict):
            raise RuntimeError("GitHub pull request was not found.")

        pull_request = serialize_pull_request(payload)
        if pull_request is None:
            raise RuntimeError("GitHub returned an incomplete pull request.")

        return {
            "repository": f"{scope['owner']}/{scope['repo']}",
            "pull_request": pull_request,
        }

    if method == "get_files":
        payload = await github_json(
            f"/repos/{scope['owner']}/{scope['repo']}/pulls/{pull_number}/files"
            f"?page={page}&per_page={per_page}",
        )

        if not isinstance(payload, list):
            raise RuntimeError("GitHub returned an unexpected PR files response.")

        return {
            "repository": f"{scope['owner']}/{scope['repo']}",
            "pull_number": pull_number,
            "files": [
                {
                    "filename": read_string(item.get("filename")),
                    "status": read_string(item.get("status")),
                    "additions": read_number(item.get("additions")),
                    "deletions": read_number(item.get("deletions")),
                    "changes": read_number(item.get("changes")),
                    "url": read_string(item.get("blob_url")),
                }
                for item in payload
                if isinstance(item, dict)
            ],
        }

    if method == "get_comments":
        payload = await github_json(
            f"/repos/{scope['owner']}/{scope['repo']}/issues/{pull_number}/comments"
            f"?page={page}&per_page={per_page}",
        )

        if not isinstance(payload, list):
            raise RuntimeError("GitHub returned an unexpected PR comments response.")

        return {
            "repository": f"{scope['owner']}/{scope['repo']}",
            "pull_number": pull_number,
            "comments": [
                {
                    "id": read_number(item.get("id")),
                    "url": read_string(item.get("html_url")),
                    "author": read_user(item.get("user")),
                    "created_at": read_string(item.get("created_at")),
                    "updated_at": read_string(item.get("updated_at")),
                    "body": compact_text(item.get("body")),
                }
                for item in payload
                if isinstance(item, dict)
            ],
        }

    diff = await github_text(
        f"/repos/{scope['owner']}/{scope['repo']}/pulls/{pull_number}",
        "application/vnd.github.v3.diff",
    )

    return {
        "repository": f"{scope['owner']}/{scope['repo']}",
        "pull_number": pull_number,
        "diff": diff,
    }


def main() -> None:
    """Run the scoped GitHub MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
