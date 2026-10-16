import { ChatMessageRole, ChatMessageStatus, prisma } from "@repo/db";
import type { NextRequest } from "next/server";
import { deleteConversationForUser } from "../../../../../../lib/deletion";
import {
  findConnectedRepositoryForUser,
  requireRepoMindUser,
} from "../../../../../../lib/github-public";

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readMessage(value: unknown) {
  if (!isRecord(value)) {
    return null;
  }

  const message = typeof value.message === "string" ? value.message.trim() : "";
  const conversationId =
    typeof value.conversation_id === "string"
      ? value.conversation_id.trim()
      : null;

  if (message.length === 0 || message.length > 4000) {
    return null;
  }

  return {
    message,
    conversationId:
      conversationId && conversationId.length > 0 ? conversationId : null,
  };
}

function serializeMessage(message: {
  id: string;
  role: ChatMessageRole;
  content: string;
  status: ChatMessageStatus;
  errorMessage: string | null;
  createdAt: Date;
  citations: {
    id: string;
    sourceIndex: number;
    path: string;
    startLine: number;
    endLine: number;
    url: string | null;
  }[];
}) {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    status: message.status,
    error_message: message.errorMessage,
    created_at: message.createdAt.toISOString(),
    citations: message.citations.map((citation) => ({
      id: citation.id,
      source_index: citation.sourceIndex,
      path: citation.path,
      start_line: citation.startLine,
      end_line: citation.endLine,
      url: citation.url,
    })),
  };
}

async function requireRepository(owner: string, repo: string) {
  const user = await requireRepoMindUser();
  const repository = await findConnectedRepositoryForUser(user.id, owner, repo);

  if (!repository) {
    return null;
  }

  return { user, repository };
}

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ owner: string; repo: string }> },
) {
  const { owner, repo } = await context.params;
  const scope = await requireRepository(owner, repo);

  if (!scope) {
    return Response.json({ error: "Repository not found." }, { status: 404 });
  }

  const conversations = await prisma.chatConversation.findMany({
    where: {
      userId: scope.user.id,
      repositoryId: scope.repository.id,
    },
    orderBy: { updatedAt: "desc" },
    take: 20,
    include: {
      messages: {
        orderBy: { createdAt: "asc" },
        take: 100,
        include: {
          citations: {
            orderBy: { sourceIndex: "asc" },
          },
        },
      },
    },
  });

  return Response.json({
    repository: {
      id: scope.repository.id,
      owner: scope.repository.owner,
      name: scope.repository.name,
      status: scope.repository.indexStatus,
      indexed_chunk_count: scope.repository.indexedChunkCount,
    },
    conversations: conversations.map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      created_at: conversation.createdAt.toISOString(),
      updated_at: conversation.updatedAt.toISOString(),
      messages: conversation.messages.map(serializeMessage),
    })),
  });
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ owner: string; repo: string }> },
) {
  const requestId = crypto.randomUUID();
  console.info(
    JSON.stringify({
      request_id: requestId,
      stage: "request",
      operation: "chat",
      status: "started",
    }),
  );
  const { owner, repo } = await context.params;
  console.info(
    JSON.stringify({
      request_id: requestId,
      stage: "database",
      operation: "chat_prepare",
      status: "started",
    }),
  );
  const scope = await requireRepository(owner, repo);

  if (!scope) {
    return Response.json({ error: "Repository not found." }, { status: 404 });
  }

  if (
    scope.repository.indexStatus !== "READY" ||
    scope.repository.indexedChunkCount <= 0
  ) {
    return Response.json(
      { error: "Repository must finish indexing before chat is available." },
      { status: 409 },
    );
  }

  const payload = readMessage(await request.json());

  if (!payload) {
    return Response.json(
      { error: "Message is required and must be at most 4000 characters." },
      { status: 400 },
    );
  }

  let conversation = payload.conversationId
    ? await prisma.chatConversation.findFirst({
        where: {
          id: payload.conversationId,
          userId: scope.user.id,
          repositoryId: scope.repository.id,
        },
      })
    : null;

  if (payload.conversationId && !conversation) {
    return Response.json({ error: "Conversation not found." }, { status: 404 });
  }

  conversation ??= await prisma.chatConversation.create({
    data: {
      userId: scope.user.id,
      repositoryId: scope.repository.id,
      title: payload.message.slice(0, 80),
    },
  });

  const userMessage = await prisma.chatMessage.create({
    data: {
      conversationId: conversation.id,
      role: ChatMessageRole.USER,
      content: payload.message,
      status: ChatMessageStatus.SUCCEEDED,
    },
  });

  await prisma.chatConversation.update({
    where: { id: conversation.id },
    data: { updatedAt: new Date() },
  });
  console.info(
    JSON.stringify({
      request_id: requestId,
      stage: "database",
      operation: "chat_prepare",
      status: "completed",
    }),
  );

  const apiBaseUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8137";
  const credential = process.env.INTERNAL_API_CREDENTIAL;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-RepoMind-Request-Id": requestId,
  };

  if (credential) {
    headers["X-RepoMind-Internal-Credential"] = credential;
  }

  const response = await fetch(
    `${apiBaseUrl}/api/v1/repositories/chat/stream`,
    {
      method: "POST",
      signal: request.signal,
      headers,
      body: JSON.stringify({
        repository_id: scope.repository.id,
        conversation_id: conversation.id,
        user_id: scope.user.id,
        message_id: userMessage.id,
      }),
    },
  );

  if (!response.ok || !response.body) {
    const detail = await response.text();
    return Response.json(
      {
        error: `FastAPI chat failed: ${response.status}${
          detail ? ` ${detail}` : ""
        }`,
      },
      { status: 502 },
    );
  }

  return new Response(response.body, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "text/event-stream",
      "X-RepoMind-Conversation-Id": conversation.id,
      "X-RepoMind-User-Message-Id": userMessage.id,
      "X-RepoMind-Request-Id": requestId,
    },
  });
}

/** Delete an owned conversation and its messages after any response has finished. */
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ owner: string; repo: string }> },
) {
  const { owner, repo } = await context.params;
  const scope = await requireRepository(owner, repo);
  if (!scope)
    return Response.json({ error: "Repository not found." }, { status: 404 });
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  if (
    !isRecord(body) ||
    typeof body.conversation_id !== "string" ||
    body.conversation_id.trim().length === 0 ||
    body.conversation_id.length > 128
  ) {
    return Response.json(
      { error: "A valid conversation ID is required." },
      { status: 400 },
    );
  }
  try {
    console.info(
      JSON.stringify({
        stage: "database",
        operation: "chat_delete",
        status: "started",
      }),
    );
    const result = await deleteConversationForUser(
      scope.user.id,
      scope.repository.id,
      body.conversation_id.trim(),
    );
    console.info(
      JSON.stringify({
        stage: "database",
        operation: "chat_delete",
        status: result,
      }),
    );
    if (result === "not_found")
      return Response.json(
        { error: "Conversation not found." },
        { status: 404 },
      );
    if (result === "busy")
      return Response.json(
        { error: "Wait for the response to finish before deleting this chat." },
        { status: 409 },
      );
    return new Response(null, { status: 204 });
  } catch (error) {
    console.error(
      JSON.stringify({
        stage: "database",
        operation: "chat_delete",
        status: "failed",
        error_type: error instanceof Error ? error.name : "UnknownError",
      }),
    );
    return Response.json(
      { error: "Chat deletion failed. Try again." },
      { status: 500 },
    );
  }
}
