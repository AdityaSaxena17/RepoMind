import { auth } from "@clerk/nextjs/server";
import { prisma } from "@repo/db";
import { Bot, GitBranch, SlidersHorizontal, UserRound } from "lucide-react";
import { AppShell } from "../../components/app-shell";
import { repositories } from "../../lib/prototype-data";

export default async function SettingsPage() {
  const { userId } = await auth.protect();
  const user = await prisma.user.findUnique({
    where: {
      clerkId: userId,
    },
  });

  return (
    <AppShell activeItem="settings">
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">Workspace</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Settings
        </h1>
        <div className="mt-8 grid gap-4 xl:grid-cols-2">
          <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
            <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
              <UserRound className="h-4 w-4 text-sky-300" />
              Connected account
            </p>
            <p className="mt-4 text-sm text-zinc-400">
              {user?.email ?? user?.username ?? userId}
            </p>
            <p className="mt-2 text-xs text-zinc-600">
              Clerk remains the source of truth for sign-in.
            </p>
          </section>
          <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
            <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
              <GitBranch className="h-4 w-4 text-zinc-300" />
              Installed repositories
            </p>
            <p className="mt-4 text-3xl font-semibold text-zinc-50">
              {repositories.length}
            </p>
            <p className="mt-2 text-xs text-zinc-600">
              Prototype repositories represent GitHub App installations.
            </p>
          </section>
          <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
            <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
              <Bot className="h-4 w-4 text-emerald-300" />
              AI stack
            </p>
            <p className="mt-4 text-sm leading-6 text-zinc-400">
              OpenAI chat and embeddings with pgvector-backed retrieval are the
              selected backend stack for the later implementation.
            </p>
          </section>
          <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
            <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
              <SlidersHorizontal className="h-4 w-4 text-amber-300" />
              AI preferences
            </p>
            <div className="mt-4 space-y-3">
              <label className="flex items-center justify-between gap-4 text-sm text-zinc-400">
                Ground every response in citations
                <input checked readOnly type="checkbox" />
              </label>
              <label className="flex items-center justify-between gap-4 text-sm text-zinc-400">
                Show execution timeline
                <input checked readOnly type="checkbox" />
              </label>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
