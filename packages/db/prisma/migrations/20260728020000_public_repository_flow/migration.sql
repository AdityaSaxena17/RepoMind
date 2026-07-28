ALTER TABLE "Repository" DROP CONSTRAINT IF EXISTS "Repository_githubInstallationId_fkey";
ALTER TABLE "UserGitHubInstallation" DROP CONSTRAINT IF EXISTS "UserGitHubInstallation_githubInstallationId_fkey";
ALTER TABLE "UserGitHubInstallation" DROP CONSTRAINT IF EXISTS "UserGitHubInstallation_userId_fkey";

DROP INDEX IF EXISTS "Repository_githubInstallationId_idx";
DROP INDEX IF EXISTS "GitHubInstallation_accountLogin_idx";
DROP INDEX IF EXISTS "GitHubInstallation_githubInstallationId_key";
DROP INDEX IF EXISTS "GitHubOAuthState_expiresAt_idx";
DROP INDEX IF EXISTS "GitHubOAuthState_stateHash_key";
DROP INDEX IF EXISTS "GitHubOAuthState_userId_idx";
DROP INDEX IF EXISTS "UserGitHubInstallation_githubInstallationId_idx";
DROP INDEX IF EXISTS "UserGitHubInstallation_userId_githubInstallationId_key";

ALTER TABLE "Repository" DROP COLUMN IF EXISTS "githubInstallationId";

DROP TABLE IF EXISTS "UserGitHubInstallation";
DROP TABLE IF EXISTS "GitHubOAuthState";
DROP TABLE IF EXISTS "GitHubInstallation";
