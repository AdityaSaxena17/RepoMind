import { RepositoryIndexJobTrigger } from "@repo/db";
import type { NextRequest } from "next/server";
import {
  connectRepositoryForUser,
  readNumberLike,
  requireRepoMindUser,
} from "../../../../lib/github-app";

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readPayload(body: unknown) {
  if (!isRecord(body)) {
    return null;
  }

  const installationId = readNumberLike(body.installation_id);
  const repositoryId = readNumberLike(body.github_repository_id);

  if (!installationId || !repositoryId) {
    return null;
  }

  return { installationId, repositoryId };
}

async function enqueueFastApiJob(
  repositoryId: string,
  jobId: string,
  userId: string,
) {
  const apiBaseUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
  const credential = process.env.INTERNAL_API_CREDENTIAL;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-RepoMind-User-Id": userId,
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
    throw new Error(`FastAPI repository enqueue failed: ${response.status}`);
  }
}

export async function POST(request: NextRequest) {
  const user = await requireRepoMindUser();
  const payload = readPayload(await request.json());

  if (!payload) {
    return Response.json(
      { error: "installation_id and github_repository_id are required strings." },
      { status: 400 },
    );
  }

  try {
    const { repository, job, status } = await connectRepositoryForUser(
      user.id,
      payload.installationId,
      payload.repositoryId,
      RepositoryIndexJobTrigger.CONNECT,
    );

    await enqueueFastApiJob(repository.id, job.id, user.id);

    return Response.json({
      repository_id: repository.id,
      job_id: job.id,
      status,
    });
  } catch (error) {
    console.error("Repository connection failed", error);

    return Response.json(
      { error: "Repository connection failed." },
      { status: 400 },
    );
  }
}
