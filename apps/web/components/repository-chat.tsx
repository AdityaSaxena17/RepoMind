"use client";

import { ArrowUp, Bot, Loader2 } from "lucide-react";
import { useState } from "react";

interface ChatMessage {
  role: "assistant" | "user";
  content: string;
}

export function RepositoryChat({
  owner,
  repo,
}: {
  owner: string;
  repo: string;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [toolActivity, setToolActivity] = useState<string | null>(null);

  async function sendMessage() {
    const content = input.trim();

    if (!content || loading) {
      return;
    }

    const nextMessages: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setToolActivity(null);

    const response = await fetch(
      `/api/repositories/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/chat`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: nextMessages.slice(-20),
        }),
      },
    );

    if (!response.ok || !response.body) {
      setMessages([
        ...nextMessages,
        { role: "assistant", content: "Chat request failed." },
      ]);
      setLoading(false);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let assistant = "";

    setMessages([...nextMessages, { role: "assistant", content: "" }]);

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const event of events) {
        const dataLine = event
          .split("\n")
          .find((line) => line.startsWith("data: "));

        if (!dataLine) {
          continue;
        }

        const raw = dataLine.slice(6);

        if (raw === "[DONE]") {
          continue;
        }

        try {
          const parsed = JSON.parse(raw) as {
            delta?: string;
            type?: string;
            item?: { name?: string; type?: string };
          };

          if (
            parsed.type === "response.output_text.delta" &&
            typeof parsed.delta === "string"
          ) {
            assistant += parsed.delta;
            setMessages([
              ...nextMessages,
              { role: "assistant", content: assistant },
            ]);
          }

          if (
            parsed.type === "response.output_item.added" &&
            parsed.item?.type === "mcp_call" &&
            parsed.item.name
          ) {
            setToolActivity(parsed.item.name);
          }
        } catch {
          setToolActivity("Reading GitHub context");
        }
      }
    }

    setLoading(false);
  }

  return (
    <section className="flex min-h-[calc(100svh-4rem)] min-w-0 flex-col">
      <div className="border-b border-white/10 px-5 py-5 md:px-8">
        <p className="text-sm text-zinc-500">
          {owner}/{repo}
        </p>
        <h1 className="mt-2 text-2xl font-semibold tracking-normal text-zinc-50">
          AI Codebase Chat
        </h1>
      </div>
      <div className="flex-1 space-y-5 px-5 py-6 md:px-8">
        {messages.length === 0 ? (
          <div className="rounded-lg border border-white/10 bg-white/[0.03] p-5 text-sm leading-6 text-zinc-400">
            Ask about files, branches, commits, issues, or pull requests. This
            uses live GitHub MCP context and does not save chat history.
          </div>
        ) : null}
        {messages.map((message, index) => (
          <div
            className={`flex gap-3 ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
            key={`${message.role}-${index}`}
          >
            {message.role === "assistant" ? (
              <span className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-md border border-sky-400/30 bg-sky-400/10 text-sky-200">
                <Bot className="h-4 w-4" />
              </span>
            ) : null}
            <div
              className={`max-w-3xl rounded-lg border p-4 ${
                message.role === "user"
                  ? "border-white/10 bg-zinc-100 text-zinc-950"
                  : "border-white/10 bg-white/[0.03] text-zinc-300"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm leading-6">
                {message.content || (loading ? "Thinking..." : "")}
              </p>
            </div>
          </div>
        ))}
        {toolActivity ? (
          <p className="text-xs text-zinc-500">Tool activity: {toolActivity}</p>
        ) : null}
      </div>
      <div className="border-t border-white/10 p-4 md:px-8">
        <div className="flex min-h-14 items-center gap-3 rounded-lg border border-white/10 bg-zinc-950 px-4">
          <input
            className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void sendMessage();
              }
            }}
            placeholder="Ask about this repository..."
            value={input}
          />
          <button
            className="grid h-9 w-9 place-items-center rounded-md bg-zinc-100 text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={loading || input.trim().length === 0}
            onClick={sendMessage}
            type="button"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </section>
  );
}
