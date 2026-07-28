import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import {
  githubOAuthAuthorizeUrl,
  readNumberLike,
  validateSetupState,
} from "../../../../lib/github-app";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const installationId = readNumberLike(
    request.nextUrl.searchParams.get("installation_id"),
  );
  const state = request.nextUrl.searchParams.get("state");

  if (!installationId || !state) {
    return Response.json({ error: "Missing GitHub setup parameters." }, { status: 400 });
  }

  if (!(await validateSetupState(state, installationId))) {
    return Response.json({ error: "Invalid or expired GitHub setup state." }, { status: 400 });
  }

  return NextResponse.redirect(
    githubOAuthAuthorizeUrl(state, request.nextUrl.origin),
  );
}
