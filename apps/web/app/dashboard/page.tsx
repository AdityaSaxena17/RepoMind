import { auth } from "@clerk/nextjs/server";
import { prisma } from "@repo/db";
import { Database, GitBranch, Plus, RefreshCw } from "lucide-react";
import { AppShell } from "../../components/app-shell";
import { RepositoryCard } from "../../components/repository-card";
import { repositories } from "../../lib/prototype-data";

export default async function Dashboard() {
  const { userId } = await auth.protect();
  const user = await prisma.user.findUnique({
    where: {
      clerkId: userId,
    },
  });

  return (
    <AppShell
      activeItem="dashboard"
      rightPanel={
        <div className="space-y-6">
          <section>
            <p className="text-sm font-semibold text-zinc-100">
              GitHub connection
            </p>
            <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <GitBranch className="h-5 w-5 text-sky-300" />
              <p className="mt-3 text-sm font-medium text-zinc-100">
                GitHub App ready
              </p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">
                Prototype repositories mirror the installed-repository state the
                GitHub App will provide.
              </p>
            </div>
          </section>
          <section>
            <p className="text-sm font-semibold text-zinc-100">Account sync</p>
            <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <Database className="h-5 w-5 text-emerald-300" />
              <p className="mt-3 text-sm font-medium text-zinc-100">
                {user ? "Database user synced" : "Waiting for Clerk webhook"}
              </p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">
                {user
                  ? `Stored as ${user.email ?? user.clerkId}.`
                  : `Active Clerk user: ${userId}.`}
              </p>
            </div>
          </section>
        </div>
      }
    >
      <div className="px-5 py-6 md:px-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm font-medium text-sky-300">Dashboard</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
              Connected repositories
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
              Browse repositories connected through the RepoMind GitHub App,
              track indexing progress, and open a focused AI workspace.
            </p>
          </div>
          <div className="flex gap-2">
            <button className="inline-flex h-10 items-center gap-2 rounded-md border border-white/10 px-3 text-sm text-zinc-300">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button className="inline-flex h-10 items-center gap-2 rounded-md bg-zinc-100 px-3 text-sm font-medium text-zinc-950">
              <Plus className="h-4 w-4" />
              Connect
            </button>
          </div>
        </div>
        <div className="mt-8 grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {repositories.map((repository) => (
            <RepositoryCard
              key={`${repository.owner}/${repository.name}`}
              repository={repository}
            />
          ))}
        </div>
      </div>
    </AppShell>
  );
}
