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
import { createHash, createHmac, createSign, randomBytes } from "node:crypto";

const githubApiVersion = process.env.GITHUB_API_VERSION ?? "2022-11-28";
const githubApiBaseUrl = process.env.GITHUB_API_BASE_URL ?? "https://api.github.com";

function requiredEnv(name: string) {
  const value = process.env[name];

  if (!value || value.trim().length === 0) {
    throw new Error(`${name} is required.`);
  }

  return value.trim();
}

function optionalEnv(name: string) {
  const value = process.env[name];

  return value && value.trim().length > 0 ? value.trim() : null;
}

function requiredStateSecret() {
  const value = optionalEnv("GITHUB_OAUTH_STATE_SECRET") ??
    optionalEnv("GITHUB_APP_CLIENT_SECRET");

  if (!value) {
    throw new Error("GITHUB_OAUTH_STATE_SECRET is required.");
  }

  return value;
}

function hasGitHubAppPrivateKey() {
  return optionalEnv("GITHUB_APP_PRIVATE_KEY")?.includes("BEGIN") ?? false;
}

function base64url(value: Buffer | string) {
  return Buffer.from(value)
    .toString("base64")
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

function hashState(state: string) {
  return createHash("sha256").update(state).digest("hex");
}

function signState(nonce: string) {
  return createHmac("sha256", requiredStateSecret())
    .update(nonce)
    .digest("hex");
}

function getGitHubPrivateKey() {
  return requiredEnv("GITHUB_APP_PRIVATE_KEY").replaceAll("\\n", "\n");
}

function createGitHubAppJwt() {
  const now = Math.floor(Date.now() / 1000);
  const header = base64url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const payload = base64url(
    JSON.stringify({
      iat: now - 60,
      exp: now + 540,
      iss: requiredEnv("GITHUB_APP_ID"),
    }),
  );
  const data = `${header}.${payload}`;
  const signature = createSign("RSA-SHA256")
    .update(data)
    .sign(getGitHubPrivateKey());

  return `${data}.${base64url(signature)}`;
}

async function githubFetch(
  path: string,
  token: string,
  init?: RequestInit,
) {
  const response = await fetch(`${githubApiBaseUrl}${path}`, {
    ...init,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "RepoMind",
      "X-GitHub-Api-Version": githubApiVersion,
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`GitHub request failed: ${response.status} ${detail.slice(0, 200)}`);
  }

  return response;
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

function readRepositoryVisibility(repo: Record<string, unknown>) {
  const visibility = readString(repo.visibility);

  if (visibility === "internal") {
    return RepositoryVisibility.INTERNAL;
  }

  if (visibility === "private" || repo.private === true) {
    return RepositoryVisibility.PRIVATE;
  }

  return RepositoryVisibility.PUBLIC;
}

function readRepoOwnerLogin(repo: Record<string, unknown>) {
  const owner = repo.owner;

  if (!isRecord(owner)) {
    return null;
  }

  return readString(owner.login);
}

function readRepoSummary(repo: unknown) {
  if (!isRecord(repo)) {
    return null;
  }

  const id = readNumberLike(repo.id);
  const owner = readRepoOwnerLogin(repo);
  const name = readString(repo.name);
  const fullName = readString(repo.full_name);
  const defaultBranch = readString(repo.default_branch);

  if (!id || !owner || !name || !fullName || !defaultBranch) {
    return null;
  }

  return {
    githubRepositoryId: id,
    owner,
    name,
    fullName,
    visibility: readRepositoryVisibility(repo),
    defaultBranch,
    archived: repo.archived === true,
    htmlUrl: readString(repo.html_url),
  };
}

function readInstallationAccount(value: unknown) {
  if (!isRecord(value)) {
    return { login: "unknown", type: "unknown" };
  }

  return {
    login: readString(value.login) ?? "unknown",
    type: readString(value.type) ?? "unknown",
  };
}

export async function requireRepoMindUser() {
  const { userId } = await auth.protect();

  return prisma.user.upsert({
    where: { clerkId: userId },
    create: { clerkId: userId },
    update: {},
  });
}

export async function createInstallState(userId: string) {
  const nonce = randomBytes(32).toString("hex");
  const state = `${nonce}.${signState(nonce)}`;

  await prisma.gitHubOAuthState.create({
    data: {
      stateHash: hashState(state),
      userId,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000),
    },
  });

  return state;
}

export function githubOAuthAuthorizeUrl(state: string, origin: string) {
  const clientId = requiredEnv("GITHUB_APP_CLIENT_ID");
  const callbackUrl = new URL("/api/github/oauth/callback", origin);
  const authorizeUrl = new URL("https://github.com/login/oauth/authorize");
  authorizeUrl.searchParams.set("client_id", clientId);
  authorizeUrl.searchParams.set("state", state);
  authorizeUrl.searchParams.set("redirect_uri", callbackUrl.toString());

  return authorizeUrl;
}

export async function validateSetupState(state: string, installationId: string) {
  const [nonce, signature] = state.split(".");

  if (!nonce || !signature || signState(nonce) !== signature) {
    return false;
  }

  const record = await prisma.gitHubOAuthState.findUnique({
    where: { stateHash: hashState(state) },
  });

  if (!record || record.usedAt || record.expiresAt <= new Date()) {
    return false;
  }

  await prisma.gitHubOAuthState.update({
    where: { id: record.id },
    data: { githubInstallationId: installationId },
  });

  return true;
}

async function createInstallationToken(
  installationId: string,
  repositoryName?: string,
) {
  const body = repositoryName ? JSON.stringify({ repositories: [repositoryName] }) : undefined;
  const response = await githubFetch(
    `/app/installations/${installationId}/access_tokens`,
    createGitHubAppJwt(),
    {
      body,
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
    },
  );
  const payload = await response.json();

  if (!isRecord(payload) || typeof payload.token !== "string") {
    throw new Error("GitHub did not return an installation token.");
  }

  return payload.token;
}

export async function exchangeGitHubCode(code: string) {
  const response = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      client_id: requiredEnv("GITHUB_APP_CLIENT_ID"),
      client_secret: requiredEnv("GITHUB_APP_CLIENT_SECRET"),
      code,
    }),
  });

  if (!response.ok) {
    throw new Error("GitHub OAuth token exchange failed.");
  }

  const payload = await response.json();

  if (!isRecord(payload) || typeof payload.access_token !== "string") {
    throw new Error("GitHub OAuth token exchange did not return an access token.");
  }

  return payload.access_token;
}

export async function fetchGitHubAppSlug() {
  if (!hasGitHubAppPrivateKey()) {
    return null;
  }

  const response = await githubFetch("/app", createGitHubAppJwt());
  const payload = await response.json();

  if (!isRecord(payload)) {
    return null;
  }

  return readString(payload.slug);
}

export async function linkVerifiedInstallation(state: string, userToken: string) {
  const record = await prisma.gitHubOAuthState.findUnique({
    where: { stateHash: hashState(state) },
  });

  if (!record || record.usedAt || record.expiresAt <= new Date()) {
    throw new Error("GitHub OAuth state is invalid or expired.");
  }

  const response = await githubFetch("/user/installations?per_page=100", userToken);
  const payload = await response.json();
  const installations = isRecord(payload) && Array.isArray(payload.installations)
    ? payload.installations
    : [];
  const matches = installations.filter((installation) => {
    if (!isRecord(installation)) {
      return false;
    }

    return record.githubInstallationId
      ? readNumberLike(installation.id) === record.githubInstallationId
      : readNumberLike(installation.id) !== null;
  });

  if (matches.length === 0 && record.githubInstallationId) {
    throw new Error("GitHub installation is not accessible to this GitHub user.");
  }

  if (matches.length === 0) {
    await prisma.gitHubOAuthState.update({
      where: { id: record.id },
      data: { usedAt: new Date() },
    });

    return 0;
  }

  const repoMindUser = await prisma.user.findUniqueOrThrow({
    where: { id: record.userId },
  });

  for (const match of matches) {
    if (!isRecord(match)) {
      continue;
    }

    const installationId = readNumberLike(match.id);

    if (!installationId) {
      continue;
    }

    const account = readInstallationAccount(match.account);

    await prisma.$transaction([
      prisma.gitHubInstallation.upsert({
        where: { githubInstallationId: installationId },
        create: {
          githubInstallationId: installationId,
          accountLogin: account.login,
          accountType: account.type,
          targetType: readString(match.target_type),
          suspendedAt: match.suspended_at ? new Date(String(match.suspended_at)) : null,
        },
        update: {
          accountLogin: account.login,
          accountType: account.type,
          targetType: readString(match.target_type),
          suspendedAt: match.suspended_at ? new Date(String(match.suspended_at)) : null,
        },
      }),
      prisma.userGitHubInstallation.upsert({
        where: {
          userId_githubInstallationId: {
            userId: repoMindUser.id,
            githubInstallationId: installationId,
          },
        },
        create: {
          userId: repoMindUser.id,
          githubInstallationId: installationId,
        },
        update: {},
      }),
    ]);
  }

  await prisma.$transaction([
    prisma.gitHubOAuthState.update({
      where: { id: record.id },
      data: { usedAt: new Date() },
    }),
  ]);

  return matches.length;
}

export async function syncUserRepositories(userId: string) {
  const installations = await prisma.userGitHubInstallation.findMany({
    where: { userId },
    include: { installation: true },
    orderBy: { createdAt: "desc" },
  });
  const repositories: Prisma.RepositoryGetPayload<{
    include: {
      users: true;
    };
  }>[] = [];

  for (const installation of installations) {
    const token = await createInstallationToken(
      installation.githubInstallationId,
    );
    let page = 1;

    while (page <= 20) {
      const response = await githubFetch(
        `/installation/repositories?per_page=100&page=${page}`,
        token,
      );
      const payload = await response.json();
      const items = isRecord(payload) && Array.isArray(payload.repositories)
        ? payload.repositories
        : [];

      if (items.length === 0) {
        break;
      }

      for (const item of items) {
        const summary = readRepoSummary(item);

        if (!summary) {
          continue;
        }

        const repository = await prisma.repository.upsert({
          where: { githubRepositoryId: summary.githubRepositoryId },
          create: {
            ...summary,
            githubInstallationId: installation.githubInstallationId,
            accessStatus: RepositoryAccessStatus.ACTIVE,
          },
          update: {
            ...summary,
            githubInstallationId: installation.githubInstallationId,
            accessStatus: RepositoryAccessStatus.ACTIVE,
          },
          include: {
            users: {
              where: {
                userId,
                accessStatus: RepositoryAccessStatus.ACTIVE,
              },
            },
          },
        });
        repositories.push(repository);
      }

      if (items.length < 100) {
        break;
      }

      page += 1;
    }
  }

  return repositories;
}

export async function connectRepositoryForUser(
  userId: string,
  installationId: string,
  repositoryId: string,
  trigger: RepositoryIndexJobTrigger,
) {
  const userInstallation = await prisma.userGitHubInstallation.findUnique({
    where: {
      userId_githubInstallationId: {
        userId,
        githubInstallationId: installationId,
      },
    },
  });

  if (!userInstallation) {
    throw new Error("GitHub installation is not linked to this user.");
  }

  const token = await createInstallationToken(installationId);
  const response = await githubFetch(`/repositories/${repositoryId}`, token);
  const summary = readRepoSummary(await response.json());

  if (!summary) {
    throw new Error("GitHub repository metadata response was incomplete.");
  }

  const repository = await prisma.repository.upsert({
    where: { githubRepositoryId: repositoryId },
    create: {
      ...summary,
      githubInstallationId: installationId,
      accessStatus: RepositoryAccessStatus.ACTIVE,
      indexStatus: RepositoryIndexStatus.NOT_INDEXED,
    },
    update: {
      ...summary,
      githubInstallationId: installationId,
      accessStatus: RepositoryAccessStatus.ACTIVE,
      lastIndexError: null,
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

  const existingJob = await prisma.repositoryIndexJob.findFirst({
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
    return { repository, job: existingJob, status: repository.indexStatus };
  }

  const nextStatus = repository.indexStatus === RepositoryIndexStatus.READY
    ? RepositoryIndexStatus.UPDATING
    : RepositoryIndexStatus.INDEXING;
  const head = await githubFetch(
    `/repos/${summary.owner}/${summary.name}/commits/${encodeURIComponent(summary.defaultBranch)}`,
    token,
  );
  const headPayload = await head.json();
  const targetCommitSha = isRecord(headPayload) ? readString(headPayload.sha) : null;

  const job = await prisma.$transaction(async (tx) => {
    await tx.repository.update({
      where: { id: repository.id },
      data: {
        indexStatus: nextStatus,
        indexProgress: 0,
        targetCommitSha,
        lastIndexError: null,
      },
    });

    return tx.repositoryIndexJob.create({
      data: {
        repositoryId: repository.id,
        trigger,
        status: RepositoryIndexJobStatus.QUEUED,
        sourceCommitSha: repository.indexedCommitSha,
        targetCommitSha,
      },
    });
  }, {
    isolationLevel: Prisma.TransactionIsolationLevel.ReadCommitted,
  });

  return { repository, job, status: nextStatus };
}

export function repositoryRoute(repository: { owner: string; name: string }) {
  return `/repositories/${encodeURIComponent(repository.owner)}/${encodeURIComponent(repository.name)}`;
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

export async function githubTokenForConnectedRepository(
  userId: string,
  owner: string,
  name: string,
) {
  const repository = await findConnectedRepositoryForUser(userId, owner, name);

  if (!repository) {
    throw new Error("Repository is not connected for this user.");
  }

  return {
    repository,
    token: await createInstallationToken(
      repository.githubInstallationId,
      repository.name,
    ),
  };
}

export {
  createInstallationToken,
  githubFetch,
  hasGitHubAppPrivateKey,
  optionalEnv,
  readNumberLike,
  readString,
};
