import {
  ArrowUp,
  Bot,
  Clock,
  FileCode2,
  GitBranch,
  MessageSquare,
} from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../components/app-shell";
import { Timeline } from "../../../../components/timeline";
import {
  conversations,
  getRepository,
  repositories,
} from "../../../../lib/prototype-data";

export default async function RepositoryWorkspace({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = getRepository(owner, repo);

  if (!repository) {
    notFound();
  }

  const conversation = conversations[0];

  if (!conversation) {
    notFound();
  }

  return (
    <AppShell
      activeItem="chat"
      repository={repository}
      rightPanel={
        <div className="space-y-7">
          <section>
            <p className="text-sm font-semibold text-zinc-100">
              Files referenced
            </p>
            <div className="mt-3 space-y-2">
              {conversation.messages.flatMap((message) =>
                (message.citations ?? []).map((citation) => (
                  <div
                    className="rounded-md border border-white/10 bg-white/[0.03] p-3"
                    key={`${citation.filePath}-${citation.startLine}`}
                  >
                    <p className="truncate text-sm font-medium text-zinc-200">
                      {citation.label}
                    </p>
                    <p className="mt-1 truncate font-mono text-xs text-sky-300">
                      {citation.filePath}:{citation.startLine}-
                      {citation.endLine}
                    </p>
                  </div>
                )),
              )}
            </div>
          </section>
          <section>
            <p className="text-sm font-semibold text-zinc-100">
              Execution timeline
            </p>
            <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <Timeline steps={conversation.timeline} />
            </div>
          </section>
        </div>
      }
    >
      <div className="grid min-h-[calc(100svh-4rem)] grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="hidden border-r border-white/10 bg-[#0c0c0f] p-4 lg:block">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-zinc-100">Chats</p>
            <MessageSquare className="h-4 w-4 text-zinc-500" />
          </div>
          <div className="mt-4 space-y-2">
            {conversations.map((item) => (
              <div
                className="rounded-md border border-white/10 bg-white/[0.04] p-3"
                key={item.id}
              >
                <p className="line-clamp-2 text-sm font-medium text-zinc-100">
                  {item.title}
                </p>
                <p className="mt-2 flex items-center gap-1 text-xs text-zinc-500">
                  <Clock className="h-3 w-3" />
                  {item.updatedAt}
                </p>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <p className="flex items-center gap-2 text-xs font-medium text-zinc-300">
              <GitBranch className="h-3.5 w-3.5 text-sky-300" />
              {repository.branch}
            </p>
            <p className="mt-2 text-xs leading-5 text-zinc-500">
              {repository.indexedFiles.toLocaleString()} indexed files and{" "}
              {repository.embeddings.toLocaleString()} embeddings.
            </p>
          </div>
        </aside>
        <section className="flex min-w-0 flex-col">
          <div className="border-b border-white/10 px-5 py-5 md:px-8">
            <p className="text-sm text-zinc-500">
              {repository.owner}/{repository.name}
            </p>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal text-zinc-50">
              AI Codebase Chat
            </h1>
          </div>
          <div className="flex-1 space-y-5 px-5 py-6 md:px-8">
            {conversation.messages.map((message, index) => (
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
                  <p className="text-sm leading-6">{message.content}</p>
                  {message.citations ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {message.citations.map((citation) => (
                        <span
                          className="inline-flex items-center gap-1 rounded-md border border-sky-400/20 bg-sky-400/10 px-2 py-1 font-mono text-xs text-sky-200"
                          key={`${citation.filePath}-${citation.startLine}`}
                        >
                          <FileCode2 className="h-3 w-3" />
                          {citation.filePath}:{citation.startLine}-
                          {citation.endLine}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
            <div className="rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <p className="text-sm font-medium text-zinc-100">
                Suggested follow-ups
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {conversation.followUps.map((followUp) => (
                  <button
                    className="rounded-md border border-white/10 px-3 py-2 text-left text-sm text-zinc-300 hover:bg-white/[0.04]"
                    key={followUp}
                  >
                    {followUp}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="border-t border-white/10 p-4 md:px-8">
            <div className="flex min-h-14 items-center gap-3 rounded-lg border border-white/10 bg-zinc-950 px-4">
              <input
                className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
                placeholder="Ask about this repository..."
              />
              <button className="grid h-9 w-9 place-items-center rounded-md bg-zinc-100 text-zinc-950">
                <ArrowUp className="h-4 w-4" />
              </button>
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}

export function generateStaticParams() {
  return repositories.map((repository) => ({
    owner: repository.owner,
    repo: repository.name,
  }));
}
