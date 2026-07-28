"use client";

import {
  ArrowUpRight,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type RepositoryStatus =
  | "FAILED"
  | "INDEXING"
  | "NOT_INDEXED"
  | "READY"
  | "UPDATING";

interface RepositorySummary {
  id: string;
  owner: string;
  name: string;
  full_name: string;
  default_branch: string;
  html_url: string | null;
  status: RepositoryStatus;
  indexed_file_count: number;
  indexed_chunk_count: number;
  last_indexed_at: string | null;
  last_index_error: string | null;
}

const statusLabels: Record<RepositoryStatus, string> = {
  FAILED: "Failed",
  INDEXING: "Indexing",
  NOT_INDEXED: "Connected",
  READY: "Ready",
  UPDATING: "Updating",
};

function readRepositories(value: unknown) {
  if (
    typeof value !== "object" ||
    value === null ||
    !Array.isArray((value as { repositories?: unknown }).repositories)
  ) {
    return [];
  }

  return (value as { repositories: RepositorySummary[] }).repositories;
}

export function RepositoryPicker() {
  const [repositories, setRepositories] = useState<RepositorySummary[]>([]);
  const [query, setQuery] = useState("");
  const [owner, setOwner] = useState("");
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyRepository, setBusyRepository] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function loadRepositories() {
    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch("/api/repositories", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      setRepositories(readRepositories(await response.json()));
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Repository list failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadRepositories();
  }, []);

  useEffect(() => {
    if (
      repositories.some((repository) =>
        ["INDEXING", "UPDATING"].includes(repository.status),
      )
    ) {
      const timer = window.setInterval(() => {
        void loadRepositories();
      }, 5000);

      return () => window.clearInterval(timer);
    }

    return undefined;
  }, [repositories]);

  const filteredRepositories = useMemo(() => {
    const needle = query.trim().toLowerCase();

    if (!needle) {
      return repositories;
    }

    return repositories.filter((repository) =>
      repository.full_name.toLowerCase().includes(needle),
    );
  }, [query, repositories]);

  async function connectRepository(retryRepository?: RepositorySummary) {
    const nextOwner = retryRepository?.owner ?? owner.trim();
    const nextRepo = retryRepository?.name ?? repo.trim();
    setMessage(null);

    if (!nextOwner || !nextRepo) {
      setMessage("Owner and repository name are required.");
      return;
    }

    if (retryRepository) {
      setBusyRepository(retryRepository.id);
    } else {
      setConnecting(true);
    }

    try {
      const response = await fetch(
        retryRepository ? "/api/repositories/reindex" : "/api/repositories/connect",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            owner: nextOwner,
            repo: nextRepo,
          }),
        },
      );

      if (!response.ok) {
        const payload = (await response.json()) as { error?: string };
        throw new Error(payload.error ?? "Repository connection failed.");
      }

      setOwner("");
      setRepo("");
      await loadRepositories();
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Repository connection failed.",
      );
    } finally {
      setConnecting(false);
      setBusyRepository(null);
    }
  }

  return (
    <section className="rounded-lg border border-white/10 bg-zinc-950/70 p-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-50">
            Connect a public repository
          </h2>
          <p className="mt-2 text-sm leading-6 text-zinc-500">
            Enter a public GitHub owner and repository name. RepoMind verifies
            it, then queues indexing.
          </p>
        </div>
        <button
          className="inline-flex h-10 w-fit items-center justify-center gap-2 rounded-md border border-white/10 px-3 text-sm text-zinc-300 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={loading}
          onClick={loadRepositories}
          type="button"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
        <label className="text-sm font-medium text-zinc-200">
          Owner / username / org
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            onChange={(event) => setOwner(event.target.value)}
            placeholder="vercel"
            value={owner}
          />
        </label>
        <label className="text-sm font-medium text-zinc-200">
          Repository name
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            onChange={(event) => setRepo(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void connectRepository();
              }
            }}
            placeholder="next.js"
            value={repo}
          />
        </label>
        <button
          className="mt-7 inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap rounded-md bg-zinc-100 px-4 text-sm font-medium text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={connecting}
          onClick={() => connectRepository()}
          type="button"
        >
          {connecting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <ShieldCheck className="h-4 w-4" />
          )}
          Connect repository
        </button>
      </div>
      <div className="mt-5 flex min-h-11 items-center gap-3 rounded-md border border-white/10 bg-white/[0.03] px-3">
        <Search className="h-4 w-4 text-zinc-500" />
        <input
          className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search connected repositories"
          value={query}
        />
      </div>
      {message ? (
        <p className="mt-4 flex items-start gap-2 rounded-md border border-rose-400/20 bg-rose-400/10 p-3 text-sm text-rose-200">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          {message}
        </p>
      ) : null}
      <div className="mt-5 grid gap-3">
        {loading && repositories.length === 0 ? (
          <div className="grid min-h-40 place-items-center rounded-lg border border-dashed border-white/10 text-sm text-zinc-500">
            <span className="inline-flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading connected repositories
            </span>
          </div>
        ) : null}
        {!loading && filteredRepositories.length === 0 ? (
          <div className="grid min-h-40 place-items-center rounded-lg border border-dashed border-white/10 text-center">
            <div>
              <Search className="mx-auto h-5 w-5 text-zinc-600" />
              <p className="mt-3 text-sm font-medium text-zinc-300">
                No connected repositories yet
              </p>
            </div>
          </div>
        ) : null}
        {filteredRepositories.map((repository) => {
          const isBusy = busyRepository === repository.id;
          const canOpen = repository.status !== "FAILED";

          return (
            <article
              className="rounded-lg border border-white/10 bg-white/[0.03] p-4"
              key={repository.id}
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div className="min-w-0">
                  <p className="text-xs text-zinc-500">
                    public · default: {repository.default_branch}
                  </p>
                  <h3 className="mt-2 truncate text-base font-semibold text-zinc-50">
                    {repository.full_name}
                  </h3>
                  {repository.last_index_error ? (
                    <p className="mt-2 line-clamp-2 text-xs text-rose-200">
                      {repository.last_index_error}
                    </p>
                  ) : null}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex h-8 items-center rounded-md border border-white/10 px-2.5 text-xs text-zinc-300">
                    {statusLabels[repository.status]}
                  </span>
                  {repository.status === "INDEXING" ||
                  repository.status === "UPDATING" ? (
                    <span className="inline-flex h-8 items-center rounded-md border border-sky-400/20 px-2.5 text-xs text-sky-200">
                      Working
                    </span>
                  ) : null}
                  {canOpen ? (
                    <a
                      className="inline-flex h-9 items-center gap-2 rounded-md border border-white/10 px-3 text-sm text-zinc-200"
                      href={`/repositories/${repository.owner}/${repository.name}`}
                    >
                      Open
                      <ArrowUpRight className="h-4 w-4" />
                    </a>
                  ) : null}
                  {repository.status === "FAILED" ? (
                    <button
                      className="inline-flex h-9 items-center gap-2 rounded-md bg-zinc-100 px-3 text-sm font-medium text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                      disabled={isBusy}
                      onClick={() => connectRepository(repository)}
                      type="button"
                    >
                      {isBusy ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4" />
                      )}
                      Retry indexing
                    </button>
                  ) : null}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
