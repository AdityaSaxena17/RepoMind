import { FileCode2, Search } from "lucide-react";
import { notFound } from "next/navigation";
import { AppShell } from "../../../../../components/app-shell";
import { getRepository, searchResults } from "../../../../../lib/prototype-data";

export default async function RepositorySearch({
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
    <AppShell activeItem="search" repository={repository}>
      <div className="px-5 py-6 md:px-8">
        <p className="text-sm font-medium text-sky-300">Natural language</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
          Code search
        </h1>
        <div className="mt-6 flex min-h-12 max-w-3xl items-center gap-3 rounded-lg border border-white/10 bg-zinc-950 px-4">
          <Search className="h-4 w-4 text-zinc-500" />
          <input
            className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            defaultValue="Find where users are authenticated"
          />
        </div>
        <div className="mt-8 space-y-3">
          {searchResults.map((result) => (
            <article
              className="rounded-lg border border-white/10 bg-white/[0.03] p-5"
              key={`${result.filePath}-${result.symbolName}`}
            >
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 font-mono text-sm text-sky-300">
                    <FileCode2 className="h-4 w-4 shrink-0" />
                    <span className="truncate">{result.filePath}</span>
                  </p>
                  <h2 className="mt-3 text-lg font-semibold text-zinc-50">
                    {result.symbolName}
                  </h2>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">
                    {result.explanation}
                  </p>
                </div>
                <span className="w-fit rounded-md border border-emerald-400/20 bg-emerald-400/10 px-2 py-1 text-xs font-medium text-emerald-200">
                  {result.confidence}% confidence
                </span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {result.citations.map((citation) => (
                  <span
                    className="rounded-md border border-white/10 px-2 py-1 font-mono text-xs text-zinc-400"
                    key={`${citation.filePath}-${citation.startLine}`}
                  >
                    {citation.label}: {citation.startLine}-{citation.endLine}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
