import { auth } from "@clerk/nextjs/server";
import {
  Prisma,
  RepositoryAccessStatus,
  RepositoryIndexJobStatus,
  RepositoryIndexJobTrigger,
  RepositoryIndexStatus,
  RepositoryVisibility,
  prisma,
} from "@repo/db";

const githubApiBaseUrl = "https://api.github.com";
const ownerPattern = /^[A-Za-z0-9_.-]+$/;
const repoPattern = /^[A-Za-z0-9_.-]+$/;

export class GitHubPublicApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "GitHubPublicApiError";
  }
}

export function readPublicRepoInput(value: unknown) {
  if (typeof value !== "object" || value === null) {
    return null;
  }

  const body = value as { owner?: unknown; repo?: unknown };
  const owner = typeof body.owner === "string" ? body.owner.trim() : "";
  const repo = typeof body.repo === "string" ? body.repo.trim() : "";

  if (
    owner.length === 0 ||
    owner.length > 100 ||
    repo.length === 0 ||
    repo.length > 100 ||
    !ownerPattern.test(owner) ||
    !repoPattern.test(repo)
  ) {
    return null;
  }

  return { owner, repo };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function readNumberLike(value: unknown) {
  if (typeof value === "number" && Number.isSafeInteger(value) && value > 0) {
    return String(value);
  }

  if (typeof value === "string" && /^[1-9][0-9]*$/.test(value)) {
    return value;
  }

  return null;
}

function readOwnerLogin(repo: Record<string, unknown>) {
  return isRecord(repo.owner) ? readString(repo.owner.login) : null;
}

function readVisibility(repo: Record<string, unknown>) {
  const visibility = readString(repo.visibility);

  if (visibility === "internal") {
    return RepositoryVisibility.INTERNAL;
  }

  if (visibility === "private" || repo.private === true) {
    return RepositoryVisibility.PRIVATE;
  }

  return RepositoryVisibility.PUBLIC;
}

function readGitHubToken() {
  const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;

  if (!token) {
    return null;
  }

  const trimmed = token.trim();
  const lower = trimmed.toLowerCase();

  if (
    trimmed.length === 0 ||
    lower.includes("your") ||
    lower.includes("replace") ||
    lower.includes("example")
  ) {
    return null;
  }

  return trimmed;
}

async function githubJson(path: string) {
  const token = readGitHubToken();
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": "RepoMind",
    "X-GitHub-Api-Version": "2022-11-28",
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${githubApiBaseUrl}${path}`, {
    headers,
    cache: "no-store",
    signal: AbortSignal.timeout(15000),
  });

  if (response.status === 404) {
    throw new Error("GitHub repository was not found or is not public.");
  }

  if (response.status === 403) {
    throw new Error("GitHub rate limit blocked this request. Try again later.");
  }

  if (response.status === 401) {
    throw new GitHubPublicApiError(
      token
        ? "GitHub rejected the configured token. Check GITHUB_TOKEN or GH_TOKEN."
        : "GitHub requires authentication for this request. Set GITHUB_TOKEN or GH_TOKEN.",
      401,
    );
  }

  if (!response.ok) {
    throw new GitHubPublicApiError(
      `GitHub request failed with status ${response.status}.`,
      response.status,
    );
  }

  const payload = await response.json();

  if (!isRecord(payload)) {
    throw new Error("GitHub returned an unexpected response.");
  }

  return payload;
}

async function fetchPublicRepository(owner: string, repo: string) {
  const payload = await githubJson(
    `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`,
  );
  const githubRepositoryId = readNumberLike(payload.id);
  const ownerLogin = readOwnerLogin(payload);
  const name = readString(payload.name);
  const fullName = readString(payload.full_name);
  const defaultBranch = readString(payload.default_branch);
  const visibility = readVisibility(payload);

  if (
    !githubRepositoryId ||
    !ownerLogin ||
    !name ||
    !fullName ||
    !defaultBranch
  ) {
    throw new Error("GitHub repository metadata was incomplete.");
  }

  if (visibility !== RepositoryVisibility.PUBLIC) {
    throw new Error("Only public repositories are supported in this version.");
  }

  if (payload.archived === true) {
    throw new Error("Archived repositories cannot be connected.");
  }

  return {
    githubRepositoryId,
    owner: ownerLogin,
    name,
    fullName,
    visibility,
    defaultBranch,
    archived: false,
    htmlUrl: readString(payload.html_url),
  };
}

async function fetchDefaultBranchHead(
  owner: string,
  repo: string,
  branch: string,
) {
  const payload = await githubJson(
    `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/commits/${encodeURIComponent(branch)}`,
  );
  const sha = readString(payload.sha);

  if (!sha) {
    throw new Error("GitHub commit response did not include a sha.");
  }

  return sha;
}

export async function requireRepoMindUser() {
  const { userId } = await auth.protect();

  return prisma.user.upsert({
    where: { clerkId: userId },
    create: { clerkId: userId },
    update: {},
  });
}

export async function listConnectedRepositories(userId: string) {
  return prisma.repository.findMany({
    where: {
      accessStatus: RepositoryAccessStatus.ACTIVE,
      users: {
        some: {
          userId,
          accessStatus: RepositoryAccessStatus.ACTIVE,
        },
      },
    },
    orderBy: [{ updatedAt: "desc" }, { owner: "asc" }, { name: "asc" }],
  });
}

export async function connectPublicRepositoryForUser(
  userId: string,
  owner: string,
  repo: string,
  trigger: RepositoryIndexJobTrigger,
) {
  const summary = await fetchPublicRepository(owner, repo);
  const repository = await prisma.repository.upsert({
    where: { githubRepositoryId: summary.githubRepositoryId },
    create: {
      ...summary,
      accessStatus: RepositoryAccessStatus.ACTIVE,
      indexStatus: RepositoryIndexStatus.NOT_INDEXED,
    },
    update: {
      ...summary,
      accessStatus: RepositoryAccessStatus.ACTIVE,
    },
  });

  await prisma.userRepository.upsert({
    where: {
      userId_repositoryId: {
        userId,
        repositoryId: repository.id,
      },
    },
    create: {
      userId,
      repositoryId: repository.id,
      accessStatus: RepositoryAccessStatus.ACTIVE,
    },
    update: {
      accessStatus: RepositoryAccessStatus.ACTIVE,
    },
  });

  const targetCommitSha = await fetchDefaultBranchHead(
    summary.owner,
    summary.name,
    summary.defaultBranch,
  );

  return prisma.$transaction(
    async (tx) => {
      // Serialize decisions with worker promotion and other users' reindex requests.
      await tx.$queryRaw`
      SELECT "id" FROM "Repository" WHERE "id" = ${repository.id} FOR UPDATE
    `;
      const current = await tx.repository.findUnique({
        where: { id: repository.id },
      });
      if (!current) {
        throw new Error("Repository is no longer available.");
      }
      const existingJob = await tx.repositoryIndexJob.findFirst({
        where: {
          repositoryId: repository.id,
          status: {
            in: [
              RepositoryIndexJobStatus.QUEUED,
              RepositoryIndexJobStatus.RUNNING,
            ],
          },
        },
        orderBy: { createdAt: "desc" },
      });
      if (existingJob) {
        return {
          repository: current,
          job: existingJob,
          status: current.indexStatus,
        };
      }
      if (current.indexedCommitSha === targetCommitSha) {
        return { repository: current, job: null, status: current.indexStatus };
      }
      const nextStatus =
        current.indexStatus === RepositoryIndexStatus.READY
          ? RepositoryIndexStatus.UPDATING
          : RepositoryIndexStatus.INDEXING;
      await tx.repository.update({
        where: { id: repository.id },
        data: {
          indexStatus: nextStatus,
          targetCommitSha,
          lastIndexError: null,
        },
      });

      const job = await tx.repositoryIndexJob.create({
        data: {
          repositoryId: repository.id,
          trigger,
          status: RepositoryIndexJobStatus.QUEUED,
          sourceCommitSha: current.indexedCommitSha,
          targetCommitSha,
        },
      });
      return { repository: current, job, status: nextStatus };
    },
    {
      isolationLevel: Prisma.TransactionIsolationLevel.ReadCommitted,
    },
  );
}

export async function findConnectedRepositoryForUser(
  userId: string,
  owner: string,
  name: string,
) {
  return prisma.repository.findFirst({
    where: {
      owner,
      name,
      accessStatus: RepositoryAccessStatus.ACTIVE,
      users: {
        some: {
          userId,
          accessStatus: RepositoryAccessStatus.ACTIVE,
        },
      },
    },
  });
}

export async function publicGithubFetch(path: string) {
  return githubJson(path);
}
