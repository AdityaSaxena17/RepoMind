import { RepositoryAccessStatus } from "@repo/db";
import { findConnectedRepositoryForUser, requireRepoMindUser } from "./github-app";

export interface RepositoryView {
  owner: string;
  name: string;
  branch: string;
  indexedFiles: number;
  embeddings: number;
  status: string;
}

export async function requireConnectedRepositoryView(owner: string, name: string) {
  const user = await requireRepoMindUser();
  const repository = await findConnectedRepositoryForUser(user.id, owner, name);

  if (!repository || repository.accessStatus !== RepositoryAccessStatus.ACTIVE) {
    return null;
  }

  return {
    owner: repository.owner,
    name: repository.name,
    branch: repository.defaultBranch,
    indexedFiles: repository.indexedFileCount,
    embeddings: repository.indexedChunkCount,
    status: repository.indexStatus,
  };
}
