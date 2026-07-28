import { NextResponse } from "next/server";
import {
  createInstallState,
  fetchGitHubAppSlug,
  requireRepoMindUser,
} from "../../../../lib/github-app";

export const runtime = "nodejs";

export async function GET() {
  const user = await requireRepoMindUser();
  let appSlug = process.env.GITHUB_APP_SLUG;

  if (!appSlug) {
    try {
      appSlug = await fetchGitHubAppSlug() ?? undefined;
    } catch (error) {
      console.error("GitHub App slug lookup failed", error);
    }
  }

  if (!appSlug) {
    return Response.json(
      {
        error: "Set GITHUB_APP_SLUG to your GitHub App slug before linking GitHub.",
      },
      { status: 400 },
    );
  }

  try {
    const state = await createInstallState(user.id);
    const url = new URL(`https://github.com/apps/${appSlug}/installations/new`);
    url.searchParams.set("state", state);

    return NextResponse.redirect(url);
  } catch (error) {
    console.error("GitHub install state creation failed", error);

    return Response.json(
      { error: "GitHub App install state could not be created." },
      { status: 400 },
    );
  }
}
