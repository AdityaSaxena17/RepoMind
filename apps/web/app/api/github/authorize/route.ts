import { NextResponse } from "next/server";
import {
  createInstallState,
  githubOAuthAuthorizeUrl,
  requireRepoMindUser,
} from "../../../../lib/github-app";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    const user = await requireRepoMindUser();
    const state = await createInstallState(user.id);
    const origin = new URL(request.url).origin;

    return NextResponse.redirect(githubOAuthAuthorizeUrl(state, origin));
  } catch (error) {
    console.error("GitHub OAuth start failed", error);

    return Response.json(
      { error: "GitHub OAuth could not be started. Check GitHub App OAuth settings." },
      { status: 400 },
    );
  }
}
