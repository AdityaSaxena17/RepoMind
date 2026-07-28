import { GitBranch, GitPullRequest, UserRound } from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../../components/app-shell";
import { Timeline } from "../../../../../components/timeline";
import { getRepository, pullRequests } from "../../../../../lib/prototype-data";

export default async function RepositoryPullRequests({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = getRepository(owner, repo);

  if (!repository) {
    notFound();
  }

  return (
    <AppShell activeItem="pull-requests" repository={repository}>
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">GitHub context</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Pull requests
        </h1>
        <div className="mt-8 grid gap-4 xl:grid-cols-2">
          {pullRequests.map((pullRequest) => (
            <article
              className="rounded-lg border border-white/10 bg-white/[0.03] p-5"
              key={pullRequest.number}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="flex items-center gap-2 text-xs text-zinc-500">
                    <GitPullRequest className="h-3.5 w-3.5" />
                    PR #{pullRequest.number}
                  </p>
                  <h2 className="mt-2 text-lg font-semibold leading-6 text-zinc-50">
                    {pullRequest.title}
                  </h2>
                </div>
                <span className="w-fit rounded-md border border-white/10 px-2 py-1 text-xs text-zinc-300">
                  {pullRequest.status}
                </span>
              </div>
              <p className="mt-4 text-sm leading-6 text-zinc-400">
                {pullRequest.summary}
              </p>
              <div className="mt-5 flex flex-wrap gap-2 text-xs text-zinc-400">
                <span className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1">
                  <UserRound className="h-3 w-3" />
                  {pullRequest.author}
                </span>
                <span className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1">
                  <GitBranch className="h-3 w-3" />
                  {pullRequest.changedFiles} changed files
                </span>
              </div>
              <div className="mt-5 rounded-lg border border-white/10 bg-zinc-950/70 p-4">
                <Timeline steps={pullRequest.timeline} />
              </div>
            </article>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
