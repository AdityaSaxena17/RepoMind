import { CircleDot, ExternalLink } from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../../components/app-shell";
import {
  GitHubPublicApiError,
  publicGithubFetch,
} from "../../../../../lib/github-public";
import { requireConnectedRepositoryView } from "../../../../../lib/repository-view";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readIssues(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.items)) {
    return [];
  }

  return value.items.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const title = typeof item.title === "string" ? item.title : null;
    const url = typeof item.html_url === "string" ? item.html_url : null;
    const state = typeof item.state === "string" ? item.state : null;
    const number = typeof item.number === "number" ? item.number : null;

    if (!title || !url || !state || number === null) {
      return [];
    }

    return [{ number, title, url, state }];
  });
}

export default async function RepositoryIssues({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = await requireConnectedRepositoryView(owner, repo);

  if (!repository) {
    notFound();
  }

  let githubError: string | null = null;
  let issues: ReturnType<typeof readIssues> = [];

  try {
    issues = readIssues(await publicGithubFetch(
      `/search/issues?q=${encodeURIComponent(`repo:${owner}/${repo} is:issue`)}&per_page=20`,
    ));
  } catch (error) {
    githubError = error instanceof GitHubPublicApiError
      ? error.message
      : "GitHub issues are temporarily unavailable.";
  }

  return (
    <AppShell activeItem="issues" repository={repository}>
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">GitHub issues</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Issues
        </h1>
        <div className="mt-8 grid gap-3">
          {githubError ? (
            <p className="rounded-lg border border-amber-400/20 bg-amber-400/10 p-5 text-sm leading-6 text-amber-100">
              {githubError}
            </p>
          ) : null}
          {issues.map((issue) => (
            <a
              className="rounded-lg border border-white/10 bg-white/[0.03] p-5 transition hover:border-white/20"
              href={issue.url}
              key={issue.number}
              rel="noreferrer"
              target="_blank"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 text-xs text-zinc-500">
                    <CircleDot className="h-3.5 w-3.5" />
                    Issue #{issue.number} · {issue.state}
                  </p>
                  <h2 className="mt-2 text-base font-semibold leading-6 text-zinc-50">
                    {issue.title}
                  </h2>
                </div>
                <ExternalLink className="h-4 w-4 shrink-0 text-zinc-500" />
              </div>
            </a>
          ))}
          {!githubError && issues.length === 0 ? (
            <p className="rounded-lg border border-white/10 bg-white/[0.03] p-5 text-sm text-zinc-500">
              No GitHub issues found.
            </p>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
