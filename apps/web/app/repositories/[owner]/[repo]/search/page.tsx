import { FileCode2, Search } from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../../components/app-shell";
import {
  githubFetch,
  githubTokenForConnectedRepository,
  readString,
  requireRepoMindUser,
} from "../../../../../lib/github-app";
import { requireConnectedRepositoryView } from "../../../../../lib/repository-view";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readSearchItems(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.items)) {
    return [];
  }

  return value.items.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const path = readString(item.path);
    const htmlUrl = readString(item.html_url);

    if (!path || !htmlUrl) {
      return [];
    }

    return [{ path, htmlUrl }];
  });
}

export default async function RepositorySearch({
  params,
  searchParams,
}: {
  params: Promise<{ owner: string; repo: string }>;
  searchParams: Promise<{ q?: string }>;
}) {
  const { owner, repo } = await params;
  const repository = await requireConnectedRepositoryView(owner, repo);

  if (!repository) {
    notFound();
  }

  const { q } = await searchParams;
  const query = typeof q === "string" && q.trim().length > 0
    ? q.trim()
    : "auth";
  const user = await requireRepoMindUser();
  const { token } = await githubTokenForConnectedRepository(user.id, owner, repo);
  const response = await githubFetch(
    `/search/code?q=${encodeURIComponent(`${query} repo:${owner}/${repo}`)}&per_page=10`,
    token,
  );
  const results = readSearchItems(await response.json());

  return (
    <AppShell activeItem="search" repository={repository}>
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">GitHub search</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Code search
        </h1>
        <form className="mt-6 flex min-h-12 max-w-3xl items-center gap-3 rounded-lg border border-white/10 bg-zinc-950 px-4">
          <Search className="h-4 w-4 text-zinc-500" />
          <input
            className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            defaultValue={query}
            name="q"
            placeholder="Search code in GitHub"
          />
        </form>
        <div className="mt-8 space-y-3">
          {results.map((result) => (
            <a
              className="block rounded-lg border border-white/10 bg-white/[0.03] p-5 transition hover:border-white/20"
              href={result.htmlUrl}
              key={`${result.path}-${result.htmlUrl}`}
              rel="noreferrer"
              target="_blank"
            >
              <p className="flex items-center gap-2 font-mono text-sm text-sky-300">
                <FileCode2 className="h-4 w-4 shrink-0" />
                <span className="truncate">{result.path}</span>
              </p>
              <p className="mt-3 text-sm leading-6 text-zinc-400">
                Matched by GitHub code search. Vector retrieval is deferred.
              </p>
            </a>
          ))}
          {results.length === 0 ? (
            <p className="rounded-lg border border-white/10 bg-white/[0.03] p-5 text-sm text-zinc-500">
              No GitHub code results for this query.
            </p>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
