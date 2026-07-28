import {
  fetchGitHubAppSlug,
  hasGitHubAppPrivateKey,
} from "../../../../lib/github-app";

export const runtime = "nodejs";

export async function GET() {
  let slug = process.env.GITHUB_APP_SLUG;

  if (!slug) {
    try {
      slug = await fetchGitHubAppSlug() ?? undefined;
    } catch (error) {
      console.error("GitHub App config slug lookup failed", error);
    }
  }

  const hasStateSecret = Boolean(
    process.env.GITHUB_OAUTH_STATE_SECRET?.trim() ||
      process.env.GITHUB_APP_CLIENT_SECRET?.trim(),
  );

  const hasPrivateKey = hasGitHubAppPrivateKey();

  return Response.json({
    ready_for_install: Boolean(slug && hasStateSecret && hasPrivateKey),
    slug_configured: Boolean(slug),
    state_secret_configured: hasStateSecret,
    app_private_key_configured: hasPrivateKey,
    message: !slug
      ? "Set GITHUB_APP_SLUG to your GitHub App slug."
      : !hasStateSecret
        ? "Set GITHUB_OAUTH_STATE_SECRET or GITHUB_APP_CLIENT_SECRET."
        : !hasPrivateKey
          ? "Set GITHUB_APP_PRIVATE_KEY to the downloaded PEM private key."
        : null,
  });
}
