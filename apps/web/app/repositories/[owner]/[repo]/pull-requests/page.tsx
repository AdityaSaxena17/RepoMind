import { ExternalLink, GitPullRequest } from "lucide-react";
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

function readPullRequests(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.items)) {
    return [];
  }

  return value.items.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const title = readString(item.title);
    const url = readString(item.html_url);
    const state = readString(item.state);
    const number = typeof item.number === "number" ? item.number : null;

    if (!title || !url || !state || number === null) {
      return [];
    }

    return [{ number, title, url, state }];
  });
}

export default async function RepositoryPullRequests({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = await requireConnectedRepositoryView(owner, repo);

  if (!repository) {
    notFound();
  }

  const user = await requireRepoMindUser();
  const { token } = await githubTokenForConnectedRepository(user.id, owner, repo);
  const response = await githubFetch(
    `/search/issues?q=${encodeURIComponent(`repo:${owner}/${repo} is:pr`)}&per_page=20`,
    token,
  );
  const pullRequests = readPullRequests(await response.json());

  return (
    <AppShell activeItem="pull-requests" repository={repository}>
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">GitHub pull requests</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Pull requests
        </h1>
        <div className="mt-8 grid gap-3">
          {pullRequests.map((pullRequest) => (
            <a
              className="rounded-lg border border-white/10 bg-white/[0.03] p-5 transition hover:border-white/20"
              href={pullRequest.url}
              key={pullRequest.number}
              rel="noreferrer"
              target="_blank"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 text-xs text-zinc-500">
                    <GitPullRequest className="h-3.5 w-3.5" />
                    PR #{pullRequest.number} · {pullRequest.state}
                  </p>
                  <h2 className="mt-2 text-base font-semibold leading-6 text-zinc-50">
                    {pullRequest.title}
                  </h2>
                </div>
                <ExternalLink className="h-4 w-4 shrink-0 text-zinc-500" />
              </div>
            </a>
          ))}
          {pullRequests.length === 0 ? (
            <p className="rounded-lg border border-white/10 bg-white/[0.03] p-5 text-sm text-zinc-500">
              No GitHub pull requests found.
            </p>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
