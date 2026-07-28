import { notFound } from "next/navigation";
import { AppShell } from "../../../../components/app-shell";
import { RepositoryChat } from "../../../../components/repository-chat";
import { requireConnectedRepositoryView } from "../../../../lib/repository-view";

export default async function RepositoryWorkspace({
  params,
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repository = await requireConnectedRepositoryView(owner, repo);

  if (!repository) {
    notFound();
  }

  return (
    <AppShell
      activeItem="chat"
      repository={repository}
      rightPanel={
        <div className="space-y-6">
          <section>
            <p className="text-sm font-semibold text-zinc-100">Index status</p>
            <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <p className="text-sm text-zinc-300">{repository.status}</p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">
                {repository.indexedFiles.toLocaleString()} indexed files and{" "}
                {repository.embeddings.toLocaleString()} chunks. Retrieval over
                pgvector is deferred.
              </p>
            </div>
          </section>
          <section>
            <p className="text-sm font-semibold text-zinc-100">Live context</p>
            <p className="mt-3 text-xs leading-5 text-zinc-500">
              Chat can use GitHub MCP for files, code search, branches, commits,
              issues, and pull requests while indexing is running.
            </p>
          </section>
        </div>
      }
    >
      <RepositoryChat owner={repository.owner} repo={repository.name} />
    </AppShell>
  );
}
