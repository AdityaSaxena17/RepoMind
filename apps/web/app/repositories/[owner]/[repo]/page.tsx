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
                {repository.embeddings.toLocaleString()} chunks. Chat uses
                parent-window hybrid retrieval over pgvector and PostgreSQL FTS.
              </p>
            </div>
          </section>
          <section>
            <p className="text-sm font-semibold text-zinc-100">Retrieval</p>
            <p className="mt-3 text-xs leading-5 text-zinc-500">
              Code chat cites indexed source windows. Issues and pull requests
              remain available from the side navigation.
            </p>
          </section>
        </div>
      }
    >
      <RepositoryChat repository={repository} />
    </AppShell>
  );
}
