import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import {
  exchangeGitHubCode,
  linkVerifiedInstallation,
} from "../../../../../lib/github-app";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code");
  const state = request.nextUrl.searchParams.get("state");

  if (!code || !state) {
    return Response.json({ error: "Missing GitHub OAuth callback parameters." }, { status: 400 });
  }

  try {
    const token = await exchangeGitHubCode(code);
    const linkedInstallations = await linkVerifiedInstallation(state, token);

    if (linkedInstallations === 0) {
      return NextResponse.redirect(
        new URL("/api/github/install", request.nextUrl.origin),
      );
    }

    const dashboardUrl = new URL("/dashboard", request.nextUrl.origin);
    dashboardUrl.searchParams.set(
      "github_installations",
      String(linkedInstallations),
    );

    return NextResponse.redirect(dashboardUrl);
  } catch (error) {
    console.error("GitHub OAuth callback failed", error);

    return Response.json({ error: "GitHub installation link failed." }, { status: 400 });
  }
}
