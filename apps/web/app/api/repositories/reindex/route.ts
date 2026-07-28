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
    throw new Error("FastAPI repository enqueue failed.");
  }
}

export async function POST(request: NextRequest) {
  const user = await requireRepoMindUser();
  const body = await request.json();

  if (!isRecord(body)) {
    return Response.json({ error: "Invalid reindex payload." }, { status: 400 });
  }

  const installationId = readNumberLike(body.installation_id);
  const repositoryId = readNumberLike(body.github_repository_id);

  if (!installationId || !repositoryId) {
    return Response.json({ error: "Missing repository identifiers." }, { status: 400 });
  }

  try {
    const { repository, job, status } = await connectRepositoryForUser(
      user.id,
      installationId,
      repositoryId,
      RepositoryIndexJobTrigger.RETRY,
    );

    await enqueueFastApiJob(repository.id, job.id, user.id);

    return Response.json({
      repository_id: repository.id,
      job_id: job.id,
      status,
    });
  } catch (error) {
    console.error("Repository reindex failed", error);

    return Response.json({ error: "Repository reindex failed." }, { status: 400 });
  }
}
