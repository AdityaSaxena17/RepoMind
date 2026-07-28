"use client";

import {
  ArrowUpRight,
  CheckCircle2,
  Loader2,
  Lock,
  RefreshCw,
  Search,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type RepositoryPickerStatus =
  | "ACCESS_REMOVED"
  | "FAILED"
  | "INDEXING"
  | "NOT_CONNECTED"
  | "NOT_INDEXED"
  | "READY"
  | "UPDATING";

interface GitHubRepositorySummary {
  id: string;
  github_repository_id: string;
  installation_id: string;
  owner: string;
  name: string;
  full_name: string;
  visibility: "PUBLIC" | "PRIVATE" | "INTERNAL";
  default_branch: string;
  archived: boolean;
  html_url: string | null;
  connected: boolean;
  status: RepositoryPickerStatus;
  progress: number;
  indexed_file_count: number;
  indexed_chunk_count: number;
  last_indexed_at: string | null;
  last_index_error: string | null;
}

interface GitHubConfigStatus {
  ready_for_install: boolean;
  message: string | null;
}

const statusLabels: Record<RepositoryPickerStatus, string> = {
  ACCESS_REMOVED: "Access removed",
  FAILED: "Failed",
  INDEXING: "Indexing",
  NOT_CONNECTED: "Available",
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

  return (value as { repositories: GitHubRepositorySummary[] }).repositories;
}

export function RepositoryPicker() {
  const [repositories, setRepositories] = useState<GitHubRepositorySummary[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyRepository, setBusyRepository] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [linkingGitHub, setLinkingGitHub] = useState(false);
  const [configStatus, setConfigStatus] = useState<GitHubConfigStatus | null>(
    null,
  );

  async function loadRepositories() {
    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch("/api/github/repositories", {
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

    void fetch("/api/github/config", { cache: "no-store" })
      .then((response) => response.json())
      .then((value: GitHubConfigStatus) => setConfigStatus(value))
      .catch(() => {
        setConfigStatus({
          ready_for_install: false,
          message: "GitHub App configuration could not be checked.",
        });
      });
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

  async function connect(repository: GitHubRepositorySummary, retry: boolean) {
    setBusyRepository(repository.github_repository_id);
    setMessage(null);

    try {
      const response = await fetch(
        retry ? "/api/repositories/reindex" : "/api/repositories/connect",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            installation_id: repository.installation_id,
            github_repository_id: repository.github_repository_id,
          }),
        },
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      await loadRepositories();
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Repository connection failed.",
      );
    } finally {
      setBusyRepository(null);
    }
  }

  async function linkGitHubApp() {
    setMessage(null);

    if (configStatus && !configStatus.ready_for_install) {
      setMessage(configStatus.message ?? "GitHub App configuration is incomplete.");
      return;
    }

    setLinkingGitHub(true);

    try {
      const response = await fetch("/api/github/authorize", {
        cache: "no-store",
        redirect: "manual",
      });

      if (response.type === "opaqueredirect" || response.status === 0) {
        window.location.href = "/api/github/install";
        return;
      }

      const location = response.headers.get("location");

      if (location) {
        window.location.href = location;
        return;
      }

      if (!response.ok) {
        const payload = (await response.json()) as { error?: string };
        throw new Error(payload.error ?? "GitHub App link failed.");
      }

      window.location.href = "/api/github/authorize";
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "GitHub App link failed.",
      );
      setLinkingGitHub(false);
    }
  }

  return (
    <section className="rounded-lg border border-white/10 bg-zinc-950/70 p-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-50">
            Available repositories
          </h2>
          <p className="mt-2 text-sm leading-6 text-zinc-500">
            GitHub search and chat use live GitHub context. Indexing writes the
            default branch to PostgreSQL/pgvector after connect.
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row">
          <button
            className="inline-flex h-10 min-w-40 items-center justify-center gap-2 whitespace-nowrap rounded-md bg-zinc-100 px-4 text-sm font-medium text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={linkingGitHub || configStatus?.ready_for_install === false}
            onClick={linkGitHubApp}
            type="button"
          >
            {linkingGitHub ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ShieldCheck className="h-4 w-4" />
            )}
            Link GitHub
          </button>
          <button
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-white/10 px-3 text-sm text-zinc-300 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={loading}
            onClick={loadRepositories}
            type="button"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </div>
      <div className="mt-5 flex min-h-11 items-center gap-3 rounded-md border border-white/10 bg-white/[0.03] px-3">
        <Search className="h-4 w-4 text-zinc-500" />
        <input
          className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search repositories"
          value={query}
        />
      </div>
      {message ? (
        <p className="mt-4 flex items-start gap-2 rounded-md border border-rose-400/20 bg-rose-400/10 p-3 text-sm text-rose-200">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          {message}
        </p>
      ) : null}
      {!message && configStatus?.message ? (
        <p className="mt-4 flex items-start gap-2 rounded-md border border-amber-300/20 bg-amber-300/10 p-3 text-sm text-amber-100">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          {configStatus.message}
        </p>
      ) : null}
      <div className="mt-5 grid gap-3">
        {loading && repositories.length === 0 ? (
          <div className="grid min-h-40 place-items-center rounded-lg border border-dashed border-white/10 text-sm text-zinc-500">
            <span className="inline-flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading GitHub repositories
            </span>
          </div>
        ) : null}
        {!loading && filteredRepositories.length === 0 ? (
          <div className="grid min-h-40 place-items-center rounded-lg border border-dashed border-white/10 text-center">
            <div>
              <Search className="mx-auto h-5 w-5 text-zinc-600" />
              <p className="mt-3 text-sm font-medium text-zinc-300">
                No repositories found
              </p>
            </div>
          </div>
        ) : null}
        {filteredRepositories.map((repository) => {
          const Icon = repository.visibility === "PRIVATE" ? Lock : ShieldCheck;
          const isBusy = busyRepository === repository.github_repository_id;
          const canOpen = repository.connected && repository.status !== "ACCESS_REMOVED";

          return (
            <article
              className="rounded-lg border border-white/10 bg-white/[0.03] p-4"
              key={`${repository.installation_id}-${repository.github_repository_id}`}
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 text-xs text-zinc-500">
                    <Icon className="h-3.5 w-3.5" />
                    {repository.visibility.toLowerCase()}
                    <span>default: {repository.default_branch}</span>
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
                      {repository.progress}%
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
                  <button
                    className="inline-flex h-9 items-center gap-2 rounded-md bg-zinc-100 px-3 text-sm font-medium text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={isBusy || repository.status === "ACCESS_REMOVED"}
                    onClick={() => connect(repository, repository.status === "FAILED")}
                    type="button"
                  >
                    {isBusy ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : repository.connected ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : (
                      <ShieldCheck className="h-4 w-4" />
                    )}
                    {repository.status === "FAILED"
                      ? "Retry indexing"
                      : repository.connected
                        ? "Connected"
                        : "Connect repository"}
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
