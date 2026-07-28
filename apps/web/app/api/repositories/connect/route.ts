import { RepositoryIndexJobTrigger } from "@repo/db";
import type { NextRequest } from "next/server";
import {
  connectPublicRepositoryForUser,
  readPublicRepoInput,
  requireRepoMindUser,
} from "../../../../lib/github-public";

export const runtime = "nodejs";

async function enqueueFastApiJob(
  repositoryId: string,
  jobId: string,
  userId: string,
) {
  const apiBaseUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8137";
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
    const detail = await response.text();
    throw new Error(
      `FastAPI repository enqueue failed: ${response.status}${
        detail ? ` ${detail}` : ""
      }`,
    );
  }
}

export async function POST(request: NextRequest) {
  const user = await requireRepoMindUser();
  const payload = readPublicRepoInput(await request.json());

  if (!payload) {
    return Response.json(
      { error: "Owner and repository name are required." },
      { status: 400 },
    );
  }

  try {
    const { repository, job, status } = await connectPublicRepositoryForUser(
      user.id,
      payload.owner,
      payload.repo,
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
      {
        error:
          error instanceof Error ? error.message : "Repository connection failed.",
      },
      { status: 400 },
    );
  }
}
