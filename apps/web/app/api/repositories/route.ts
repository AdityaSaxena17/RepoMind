import { requireRepoMindUser, listConnectedRepositories } from "../../../lib/github-public";

export const runtime = "nodejs";

export async function GET() {
  const user = await requireRepoMindUser();
  const repositories = await listConnectedRepositories(user.id);

  return Response.json({
    repositories: repositories.map((repository) => ({
      id: repository.id,
      github_repository_id: repository.githubRepositoryId,
      owner: repository.owner,
      name: repository.name,
      full_name: repository.fullName,
      visibility: repository.visibility,
      default_branch: repository.defaultBranch,
      archived: repository.archived,
      html_url: repository.htmlUrl,
      status: repository.indexStatus,
      progress: repository.indexProgress,
      indexed_file_count: repository.indexedFileCount,
      indexed_chunk_count: repository.indexedChunkCount,
      last_indexed_at: repository.lastIndexedAt?.toISOString() ?? null,
      last_index_error: repository.lastIndexError,
    })),
  });
}
