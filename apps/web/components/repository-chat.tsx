"use client";

import { Bot, ExternalLink, Loader2, Send, User } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RepositoryView } from "../lib/repository-view";

type MessageRole = "ASSISTANT" | "USER";
type MessageStatus = "FAILED" | "PENDING" | "STREAMING" | "SUCCEEDED";

interface Citation {
  id: string;
  source_index: number;
  path: string;
  start_line: number;
  end_line: number;
  url: string | null;
}

interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  error_message: string | null;
  created_at: string;
  citations: Citation[];
}

interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

interface ChatResponse {
  conversations?: Conversation[];
}

interface StreamPayload {
  message_id?: string;
  text?: string;
  message?: string;
  name?: string;
  citations?: Citation[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readConversations(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.conversations)) {
    return [];
  }

  return (value as ChatResponse).conversations ?? [];
}

function readStreamPayload(value: string) {
  try {
    const payload = JSON.parse(value) as unknown;

    return isRecord(payload) ? (payload as StreamPayload) : null;
  } catch {
    return null;
  }
}

function parseSseEvent(block: string) {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLines = block
    .split("\n")
    .filter((line) => line.startsWith("data:"));

  if (!eventLine || dataLines.length === 0) {
    return null;
  }

  return {
    event: eventLine.slice("event:".length).trim(),
    payload: readStreamPayload(
      dataLines.map((line) => line.slice("data:".length).trim()).join("\n"),
    ),
  };
}

export function RepositoryChat({ repository }: { repository: RepositoryView }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toolActivity, setToolActivity] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === activeConversationId,
      ) ?? null,
    [activeConversationId, conversations],
  );
  const messages = useMemo(
    () => activeConversation?.messages ?? [],
    [activeConversation],
  );
  const chatReady = repository.status === "READY" && repository.embeddings > 0;

  const loadConversations = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/repositories/${encodeURIComponent(repository.owner)}/${encodeURIComponent(repository.name)}/chat`,
        { cache: "no-store" },
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const nextConversations = readConversations(await response.json());
      setConversations(nextConversations);
      setActiveConversationId((current) =>
        current &&
        nextConversations.some((conversation) => conversation.id === current)
          ? current
          : (nextConversations[0]?.id ?? null),
      );
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Could not load chat history.",
      );
    } finally {
      setLoading(false);
    }
  }, [repository.name, repository.owner]);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  function updateAssistantMessage(
    messageId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) {
    setConversations((current) =>
      current.map((conversation) => {
        if (
          !conversation.messages.some((message) => message.id === messageId)
        ) {
          return conversation;
        }

        return {
          ...conversation,
          messages: conversation.messages.map((message) =>
            message.id === messageId ? updater(message) : message,
          ),
        };
      }),
    );
  }

  async function sendMessage() {
    const message = input.trim();

    if (!message || streaming || !chatReady) {
      return;
    }

    setInput("");
    setError(null);
    setToolActivity([]);
    setStreaming(true);
    const streamController = new AbortController();

    const createdAt = new Date().toISOString();
    const optimisticConversationId = activeConversation?.id ?? "pending";
    const optimisticUserId = `user-${createdAt}`;
    let assistantMessageId = `assistant-${createdAt}`;

    if (!activeConversation) {
      setConversations([
        {
          id: optimisticConversationId,
          title: message.slice(0, 80),
          created_at: createdAt,
          updated_at: createdAt,
          messages: [],
        },
      ]);
      setActiveConversationId(optimisticConversationId);
    }

    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === optimisticConversationId
          ? {
              ...conversation,
              messages: [
                ...conversation.messages,
                {
                  id: optimisticUserId,
                  role: "USER",
                  content: message,
                  status: "SUCCEEDED",
                  error_message: null,
                  created_at: createdAt,
                  citations: [],
                },
                {
                  id: assistantMessageId,
                  role: "ASSISTANT",
                  content: "",
                  status: "STREAMING",
                  error_message: null,
                  created_at: createdAt,
                  citations: [],
                },
              ],
            }
          : conversation,
      ),
    );

    try {
      const response = await fetch(
        `/api/repositories/${encodeURIComponent(repository.owner)}/${encodeURIComponent(repository.name)}/chat`,
        {
          method: "POST",
          signal: streamController.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversation_id: activeConversation?.id,
            message,
          }),
        },
      );

      if (!response.ok || !response.body) {
        const payload = (await response.json()) as { error?: string };
        throw new Error(payload.error ?? "Chat request failed.");
      }

      const realConversationId = response.headers.get(
        "X-RepoMind-Conversation-Id",
      );

      if (
        realConversationId &&
        realConversationId !== optimisticConversationId
      ) {
        setActiveConversationId(realConversationId);
        setConversations((current) =>
          current.map((conversation) =>
            conversation.id === optimisticConversationId
              ? { ...conversation, id: realConversationId }
              : conversation,
          ),
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      for (;;) {
        const { value, done } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          const parsed = parseSseEvent(block);

          if (!parsed || !parsed.payload) {
            continue;
          }

          if (
            parsed.event === "assistant_message" &&
            typeof parsed.payload.message_id === "string"
          ) {
            const realAssistantId = parsed.payload.message_id;
            const previousAssistantId = assistantMessageId;
            assistantMessageId = realAssistantId;
            updateAssistantMessage(previousAssistantId, (assistant) => ({
              ...assistant,
              id: realAssistantId,
            }));
          }

          if (
            parsed.event === "delta" &&
            typeof parsed.payload.text === "string"
          ) {
            updateAssistantMessage(assistantMessageId, (assistant) => ({
              ...assistant,
              content: assistant.content + parsed.payload?.text,
            }));
          }

          if (
            parsed.event === "citations" &&
            Array.isArray(parsed.payload.citations)
          ) {
            updateAssistantMessage(assistantMessageId, (assistant) => ({
              ...assistant,
              citations: parsed.payload?.citations ?? [],
            }));
          }

          if (
            parsed.event === "tool" &&
            typeof parsed.payload.name === "string"
          ) {
            const toolName = parsed.payload.name;
            setToolActivity((current) =>
              current.includes(toolName)
                ? current
                : [...current, toolName].slice(-4),
            );
          }

          if (parsed.event === "error") {
            throw new Error(parsed.payload.message ?? "Chat failed.");
          }
          if (
            parsed.event === "done" &&
            parsed.payload.message_id === assistantMessageId
          ) {
            completed = true;
          }
        }
      }

      if (!completed) {
        throw new Error(
          "Answer stream ended before completion. Partial text is unverified.",
        );
      }
      updateAssistantMessage(assistantMessageId, (assistant) => ({
        ...assistant,
        status: "SUCCEEDED",
      }));
      await loadConversations();
    } catch (sendError) {
      const messageText =
        sendError instanceof Error ? sendError.message : "Chat request failed.";
      setError(messageText);
      updateAssistantMessage(assistantMessageId, (assistant) => ({
        ...assistant,
        status: "FAILED",
        error_message: messageText,
        citations: [],
      }));
    } finally {
      streamController.abort();
      setStreaming(false);
    }
  }

  return (
    <section className="flex min-h-[calc(100svh-4rem)] min-w-0 flex-col">
      <div className="border-b border-white/10 px-5 py-5 md:px-8">
        <p className="text-sm text-zinc-500">
          {repository.owner}/{repository.name}
        </p>
        <h1 className="mt-2 text-2xl font-semibold tracking-normal text-zinc-50">
          AI Codebase Chat
        </h1>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <aside className="border-b border-white/10 p-4 lg:border-b-0 lg:border-r">
          <button
            className="w-full rounded-md border border-white/10 px-3 py-2 text-left text-sm text-zinc-200 hover:border-white/20"
            onClick={() => setActiveConversationId(null)}
            type="button"
          >
            New chat
          </button>
          <div className="mt-4 space-y-2">
            {conversations.map((conversation) => (
              <button
                className={`w-full truncate rounded-md px-3 py-2 text-left text-sm ${
                  conversation.id === activeConversation?.id
                    ? "bg-white/10 text-zinc-50"
                    : "text-zinc-500 hover:bg-white/[0.04] hover:text-zinc-200"
                }`}
                key={conversation.id}
                onClick={() => setActiveConversationId(conversation.id)}
                type="button"
              >
                {conversation.title}
              </button>
            ))}
          </div>
        </aside>
        <div className="flex min-h-0 flex-col">
          <div className="flex-1 space-y-4 overflow-y-auto px-5 py-6 md:px-8">
            {!chatReady ? (
              <div className="max-w-2xl rounded-lg border border-amber-400/20 bg-amber-400/10 p-5 text-sm leading-6 text-amber-100">
                Repository chat is available after indexing succeeds. Current
                status: {repository.status}; indexed chunks:{" "}
                {repository.embeddings}.
              </div>
            ) : null}
            {loading ? (
              <p className="flex items-center gap-2 text-sm text-zinc-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading chat history...
              </p>
            ) : null}
            {messages.length === 0 && !loading ? (
              <div className="max-w-2xl rounded-lg border border-white/10 bg-white/[0.03] p-5">
                <span className="grid h-9 w-9 place-items-center rounded-md border border-sky-400/30 bg-sky-400/10 text-sky-200">
                  <Bot className="h-4 w-4" />
                </span>
                <p className="mt-4 text-sm font-medium text-zinc-100">
                  Ask about code, issues, and pull requests.
                </p>
                <p className="mt-2 text-sm leading-6 text-zinc-500">
                  Answers use indexed code and live GitHub evidence, with
                  citations.
                </p>
              </div>
            ) : null}
            {messages.map((message) => (
              <article
                className={`rounded-lg border p-4 ${
                  message.role === "USER"
                    ? "ml-auto max-w-2xl border-sky-400/20 bg-sky-400/10"
                    : "max-w-3xl border-white/10 bg-white/[0.03]"
                }`}
                key={message.id}
              >
                <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
                  {message.role === "USER" ? (
                    <User className="h-3.5 w-3.5" />
                  ) : (
                    <Bot className="h-3.5 w-3.5" />
                  )}
                  {message.role === "USER" ? "You" : "RepoMind"}
                  {message.status === "STREAMING" ? " · streaming" : ""}
                </div>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-zinc-100">
                  {message.content || "Thinking..."}
                </p>
                {message.error_message ? (
                  <p className="mt-3 text-sm text-red-300">
                    {message.error_message}
                    {message.content ? " Partial text is unverified." : ""}
                  </p>
                ) : null}
                {message.citations.length > 0 ? (
                  <div className="mt-4 grid gap-2">
                    {message.citations.map((citation) => (
                      <a
                        className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2 text-xs text-zinc-300 hover:border-white/20"
                        href={citation.url ?? undefined}
                        key={citation.id}
                        rel="noreferrer"
                        target="_blank"
                      >
                        <span className="truncate">
                          [{citation.source_index}] {citation.path}:
                          {citation.start_line}-{citation.end_line}
                        </span>
                        {citation.url ? (
                          <ExternalLink className="h-3.5 w-3.5 shrink-0" />
                        ) : null}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
            <div ref={bottomRef} />
          </div>
          <div className="border-t border-white/10 p-4 md:px-8">
            {error ? (
              <p className="mb-3 rounded-md border border-red-400/20 bg-red-400/10 px-3 py-2 text-sm text-red-200">
                {error}
              </p>
            ) : null}
            {streaming && toolActivity.length > 0 ? (
              <p className="mb-3 rounded-md border border-sky-400/20 bg-sky-400/10 px-3 py-2 text-xs text-sky-100">
                Using GitHub MCP: {toolActivity.join(", ")}
              </p>
            ) : null}
            <div className="flex gap-3">
              <textarea
                className="min-h-12 flex-1 resize-none rounded-md border border-white/10 bg-white/[0.03] px-3 py-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 disabled:opacity-60"
                disabled={!chatReady || streaming}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder="Ask how this repository works..."
                value={input}
              />
              <button
                className="grid h-12 w-12 place-items-center rounded-md bg-zinc-100 text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={!chatReady || streaming || input.trim().length === 0}
                onClick={() => sendMessage()}
                type="button"
              >
                {streaming ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
