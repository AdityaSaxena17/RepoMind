import { RepositoryAccessStatus } from "@repo/db";
import { requireRepoMindUser, syncUserRepositories } from "../../../../lib/github-app";

export const runtime = "nodejs";

function toStatusLabel(status: string, connected: boolean, accessRemoved: boolean) {
  if (accessRemoved) {
    return "ACCESS_REMOVED";
  }

  return connected ? status : "NOT_CONNECTED";
}

export async function GET() {
  let repositories;

  try {
    const user = await requireRepoMindUser();
    repositories = await syncUserRepositories(user.id);
  } catch (error) {
    console.error("GitHub repository listing failed", error);

    return Response.json(
      {
        error:
          "Repository listing is not ready. Apply the database migrations and check GitHub App credentials.",
        repositories: [],
      },
      { status: 503 },
    );
  }

  return Response.json({
    repositories: repositories.map((repository) => {
      const connected = repository.users.some(
        (link) => link.accessStatus === RepositoryAccessStatus.ACTIVE,
      );

      return {
        id: repository.id,
        github_repository_id: repository.githubRepositoryId,
        installation_id: repository.githubInstallationId,
        owner: repository.owner,
        name: repository.name,
        full_name: repository.fullName,
        visibility: repository.visibility,
        default_branch: repository.defaultBranch,
        archived: repository.archived,
        html_url: repository.htmlUrl,
        connected,
        status: toStatusLabel(
          repository.indexStatus,
          connected,
          repository.accessStatus !== RepositoryAccessStatus.ACTIVE,
        ),
        progress: repository.indexProgress,
        indexed_file_count: repository.indexedFileCount,
        indexed_chunk_count: repository.indexedChunkCount,
        last_indexed_at: repository.lastIndexedAt?.toISOString() ?? null,
        last_index_error: repository.lastIndexError,
      };
    }),
  });
}
