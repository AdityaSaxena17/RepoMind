import { CheckCircle2, FileText, ListChecks, TriangleAlert } from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../../components/app-shell";
import { getRepository, issuePlans } from "../../../../../lib/prototype-data";

export default async function RepositoryIssues({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = getRepository(owner, repo);
  const selectedIssue = issuePlans[0];

  if (!repository || !selectedIssue) {
    notFound();
  }

  return (
    <AppShell activeItem="issues" repository={repository}>
      <div className="grid min-h-[calc(100svh-4rem)] grid-cols-1 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="border-b border-white/10 bg-[#0c0c0f] p-5 lg:border-b-0 lg:border-r">
          <p className="text-sm font-medium text-sky-300">GitHub context</p>
          <h1 className="mt-2 text-2xl font-semibold text-zinc-50">Issues</h1>
          <div className="mt-5 space-y-3">
            {issuePlans.map((issue) => (
              <article
                className={`rounded-lg border p-4 ${
                  issue.issueNumber === selectedIssue.issueNumber
                    ? "border-sky-400/30 bg-sky-400/10"
                    : "border-white/10 bg-white/[0.03]"
                }`}
                key={issue.issueNumber}
              >
                <p className="text-xs text-zinc-500">#{issue.issueNumber}</p>
                <h2 className="mt-1 text-sm font-semibold leading-5 text-zinc-100">
                  {issue.title}
                </h2>
                <p className="mt-3 text-xs text-zinc-500">{issue.status}</p>
              </article>
            ))}
          </div>
        </aside>
        <section className="min-w-0 px-5 py-6 md:px-8">
          <p className="text-sm text-zinc-500">Issue #{selectedIssue.issueNumber}</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
            {selectedIssue.title}
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-400">
            {selectedIssue.summary}
          </p>
          <div className="mt-8 grid gap-4 xl:grid-cols-2">
            <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
              <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <FileText className="h-4 w-4 text-sky-300" />
                Likely files
              </p>
              <div className="mt-4 space-y-2">
                {selectedIssue.likelyFiles.map((file) => (
                  <p
                    className="rounded-md border border-white/10 px-3 py-2 font-mono text-xs text-zinc-300"
                    key={file}
                  >
                    {file}
                  </p>
                ))}
              </div>
            </section>
            <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
              <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <ListChecks className="h-4 w-4 text-emerald-300" />
                Implementation steps
              </p>
              <ol className="mt-4 space-y-3">
                {selectedIssue.implementationSteps.map((step, index) => (
                  <li className="flex gap-3 text-sm text-zinc-400" key={step}>
                    <span className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-white/10 text-xs text-zinc-200">
                      {index + 1}
                    </span>
                    <span className="leading-6">{step}</span>
                  </li>
                ))}
              </ol>
            </section>
            <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
              <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <TriangleAlert className="h-4 w-4 text-amber-300" />
                Edge cases
              </p>
              <div className="mt-4 space-y-2">
                {selectedIssue.edgeCases.map((edgeCase) => (
                  <p className="text-sm leading-6 text-zinc-400" key={edgeCase}>
                    {edgeCase}
                  </p>
                ))}
              </div>
            </section>
            <section className="rounded-lg border border-white/10 bg-white/[0.03] p-5">
              <p className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                Testing considerations
              </p>
              <div className="mt-4 space-y-2">
                {selectedIssue.testingNotes.map((note) => (
                  <p className="text-sm leading-6 text-zinc-400" key={note}>
                    {note}
                  </p>
                ))}
              </div>
            </section>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
