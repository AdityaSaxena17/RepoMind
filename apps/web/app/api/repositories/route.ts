import {
  requireRepoMindUser,
  listConnectedRepositories,
  readPublicRepoInput,
} from "../../../lib/github-public";
import { deleteRepositoryForUser } from "../../../lib/deletion";

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
      indexed_file_count: repository.indexedFileCount,
      indexed_chunk_count: repository.indexedChunkCount,
      indexed_commit_sha: repository.indexedCommitSha,
      last_indexed_at: repository.lastIndexedAt?.toISOString() ?? null,
      last_index_error: repository.lastIndexError,
    })),
  });
}

/** Remove the caller's RepoMind connection and chats, never the GitHub repository. */
export async function DELETE(request: Request) {
  const user = await requireRepoMindUser();
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  const input = readPublicRepoInput(body);
  if (!input) {
    return Response.json(
      { error: "Valid owner and repository names are required." },
      { status: 400 },
    );
  }
  try {
    console.info(
      JSON.stringify({
        stage: "database",
        operation: "repository_delete",
        status: "started",
      }),
    );
    const result = await deleteRepositoryForUser(
      user.id,
      input.owner,
      input.repo,
    );
    console.info(
      JSON.stringify({
        stage: "database",
        operation: "repository_delete",
        status: result,
      }),
    );
    if (result === "not_found") {
      return Response.json({ error: "Repository not found." }, { status: 404 });
    }
    if (result === "busy") {
      return Response.json(
        {
          error:
            "Wait for the active chat or indexing job to finish before deleting.",
        },
        { status: 409 },
      );
    }
    return new Response(null, { status: 204 });
  } catch (error) {
    console.error(
      JSON.stringify({
        stage: "database",
        operation: "repository_delete",
        status: "failed",
        error_type: error instanceof Error ? error.name : "UnknownError",
      }),
    );
    return Response.json(
      { error: "Repository deletion failed. Try again." },
      { status: 500 },
    );
  }
}
