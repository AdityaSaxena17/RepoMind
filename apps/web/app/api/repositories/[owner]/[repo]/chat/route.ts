import type { NextRequest } from "next/server";
import {
  githubTokenForConnectedRepository,
  readString,
  requireRepoMindUser,
} from "../../../../../../lib/github-app";

export const runtime = "nodejs";

const allowedTools = [
  "get_file_contents",
  "search_code",
  "list_branches",
  "list_commits",
  "get_commit",
  "issue_read",
  "search_issues",
  "pull_request_read",
  "search_pull_requests",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readMessages(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.messages)) {
    return null;
  }

  const messages = value.messages.flatMap((message) => {
    if (!isRecord(message)) {
      return [];
    }

    const role = message.role === "user" || message.role === "assistant"
      ? message.role
      : null;
    const content = readString(message.content);

    if (!role || !content || content.length > 4000) {
      return [];
    }

    return [{ role, content }];
  });

  return messages.slice(-20);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ owner: string; repo: string }> },
) {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return Response.json({ error: "OPENAI_API_KEY is required." }, { status: 500 });
  }

  const { owner, repo } = await context.params;
  const user = await requireRepoMindUser();
  const messages = readMessages(await request.json());

  if (!messages || messages.length === 0 || messages.at(-1)?.role !== "user") {
    return Response.json({ error: "Invalid chat transcript." }, { status: 400 });
  }

  const { repository, token } = await githubTokenForConnectedRepository(
    user.id,
    owner,
    repo,
  );
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-5-mini",
      store: false,
      stream: true,
      input: [
        {
          role: "system",
          content:
            `You answer questions about ${repository.owner}/${repository.name}. ` +
            "Use GitHub MCP tools when repository facts are needed. Cite file paths " +
            "and line ranges when the tool supplies lines. Link GitHub issues and " +
            "pull requests by URL. Do not invent line numbers.",
        },
        ...messages,
      ],
      tools: [
        {
          type: "mcp",
          server_label: "github",
          server_url:
            process.env.GITHUB_MCP_SERVER_URL ??
            "https://api.githubcopilot.com/mcp/readonly",
          authorization: token,
          allowed_tools: allowedTools,
          require_approval: "never",
        },
      ],
    }),
  });

  if (!response.ok || !response.body) {
    return Response.json(
      { error: "OpenAI Responses chat failed." },
      { status: response.status },
    );
  }

  return new Response(response.body, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "text/event-stream",
    },
  });
}
