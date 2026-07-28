import {
  RepositoryAccessStatus,
  RepositoryIndexJobStatus,
  RepositoryIndexJobTrigger,
  RepositoryIndexStatus,
  prisma,
} from "@repo/db";
import type { NextRequest } from "next/server";
import { createHmac, timingSafeEqual } from "node:crypto";
import { readNumberLike, readString } from "../../../../lib/github-app";

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function verifySignature(body: string, signature: string | null) {
  const secret = process.env.GITHUB_WEBHOOK_SECRET;

  if (!secret || !signature?.startsWith("sha256=")) {
    return false;
  }

  const expected = `sha256=${createHmac("sha256", secret).update(body).digest("hex")}`;
  const expectedBuffer = Buffer.from(expected);
  const actualBuffer = Buffer.from(signature);

  return (
    expectedBuffer.length === actualBuffer.length &&
    timingSafeEqual(expectedBuffer, actualBuffer)
  );
}

async function enqueueFastApiJob(repositoryId: string, jobId: string) {
  const apiBaseUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
  const credential = process.env.INTERNAL_API_CREDENTIAL;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-RepoMind-User-Id": "github-webhook",
  };

  if (credential) {
    headers["X-RepoMind-Internal-Credential"] = credential;
  }

  const response = await fetch(`${apiBaseUrl}/api/v1/repositories/connect`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      repository_id: repositoryId,
      job_id: jobId,
    }),
  });

  if (!response.ok) {
    throw new Error("FastAPI webhook enqueue failed.");
  }
}

async function handlePush(payload: Record<string, unknown>) {
  const repositoryPayload = isRecord(payload.repository)
    ? payload.repository
    : null;
  const githubRepositoryId = repositoryPayload
    ? readNumberLike(repositoryPayload.id)
    : null;
  const targetCommitSha = readString(payload.after);

  if (!githubRepositoryId || !targetCommitSha) {
    return;
  }

  const repository = await prisma.repository.findUnique({
    where: { githubRepositoryId },
  });

  if (!repository || repository.accessStatus !== RepositoryAccessStatus.ACTIVE) {
    return;
  }

  const activeJob = await prisma.repositoryIndexJob.findFirst({
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

  if (activeJob) {
    await prisma.repository.update({
      where: { id: repository.id },
      data: { targetCommitSha },
    });
    await prisma.repositoryIndexJob.update({
      where: { id: activeJob.id },
      data: { targetCommitSha },
    });
    return;
  }

  const job = await prisma.$transaction(async (tx) => {
    await tx.repository.update({
      where: { id: repository.id },
      data: {
        targetCommitSha,
        indexStatus:
          repository.indexStatus === RepositoryIndexStatus.READY
            ? RepositoryIndexStatus.UPDATING
            : RepositoryIndexStatus.INDEXING,
        indexProgress: 0,
        lastIndexError: null,
      },
    });

    return tx.repositoryIndexJob.create({
      data: {
        repositoryId: repository.id,
        trigger: RepositoryIndexJobTrigger.PUSH,
        sourceCommitSha: repository.indexedCommitSha,
        targetCommitSha,
      },
    });
  });

  await enqueueFastApiJob(repository.id, job.id);
}

async function revokeRepository(githubRepositoryId: string) {
  const repository = await prisma.repository.findUnique({
    where: { githubRepositoryId },
  });

  if (!repository) {
    return;
  }

  await prisma.$transaction([
    prisma.repositoryChunk.deleteMany({
      where: { repositoryId: repository.id },
    }),
    prisma.repositoryFile.deleteMany({
      where: { repositoryId: repository.id },
    }),
    prisma.repositoryIndexJob.updateMany({
      where: {
        repositoryId: repository.id,
        status: {
          in: [
            RepositoryIndexJobStatus.QUEUED,
            RepositoryIndexJobStatus.RUNNING,
          ],
        },
      },
      data: {
        status: RepositoryIndexJobStatus.CANCELED,
        finishedAt: new Date(),
      },
    }),
    prisma.userRepository.updateMany({
      where: { repositoryId: repository.id },
      data: { accessStatus: RepositoryAccessStatus.REVOKED },
    }),
    prisma.repository.update({
      where: { id: repository.id },
      data: {
        accessStatus: RepositoryAccessStatus.REVOKED,
        indexStatus: RepositoryIndexStatus.NOT_INDEXED,
        indexProgress: 0,
        indexedCommitSha: null,
        indexedFileCount: 0,
        indexedChunkCount: 0,
      },
    }),
  ]);
}

async function handleRepositoryRemoval(payload: Record<string, unknown>) {
  const repositories = Array.isArray(payload.repositories_removed)
    ? payload.repositories_removed
    : [];

  for (const repository of repositories) {
    if (!isRecord(repository)) {
      continue;
    }

    const githubRepositoryId = readNumberLike(repository.id);

    if (githubRepositoryId) {
      await revokeRepository(githubRepositoryId);
    }
  }
}

export async function POST(request: NextRequest) {
  const body = await request.text();

  if (!verifySignature(body, request.headers.get("X-Hub-Signature-256"))) {
    return new Response("Invalid GitHub webhook signature.", { status: 401 });
  }

  const event = request.headers.get("X-GitHub-Event");
  const payload = JSON.parse(body);

  if (!isRecord(payload)) {
    return Response.json({ received: true, ignored: true });
  }

  if (event === "push") {
    await handlePush(payload);
  }

  if (event === "installation_repositories") {
    await handleRepositoryRemoval(payload);
  }

  if (event === "repository" && payload.action === "deleted") {
    const repository = isRecord(payload.repository) ? payload.repository : null;
    const githubRepositoryId = repository ? readNumberLike(repository.id) : null;

    if (githubRepositoryId) {
      await revokeRepository(githubRepositoryId);
    }
  }

  return Response.json({ received: true });
}
