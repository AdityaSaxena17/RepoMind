import { RepositoryAccessStatus } from "@repo/db";
import {
  findConnectedRepositoryForUser,
  requireRepoMindUser,
} from "./github-public";

export interface RepositoryView {
  id: string;
  owner: string;
  name: string;
  branch: string;
  htmlUrl: string | null;
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
    id: repository.id,
    owner: repository.owner,
    name: repository.name,
    branch: repository.defaultBranch,
    htmlUrl: repository.htmlUrl,
    indexedFiles: repository.indexedFileCount,
    embeddings: repository.indexedChunkCount,
    status: repository.indexStatus,
  };
}
