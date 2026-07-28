CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE "RepositoryVisibility" AS ENUM ('PUBLIC', 'PRIVATE', 'INTERNAL');
CREATE TYPE "RepositoryAccessStatus" AS ENUM ('ACTIVE', 'REVOKED');
CREATE TYPE "RepositoryIndexStatus" AS ENUM ('NOT_INDEXED', 'INDEXING', 'UPDATING', 'READY', 'FAILED');
CREATE TYPE "RepositoryIndexJobStatus" AS ENUM ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELED');
CREATE TYPE "RepositoryIndexJobTrigger" AS ENUM ('CONNECT', 'RETRY', 'PUSH');

CREATE TABLE "GitHubInstallation" (
    "id" TEXT NOT NULL,
    "githubInstallationId" TEXT NOT NULL,
    "accountLogin" TEXT NOT NULL,
    "accountType" TEXT NOT NULL,
    "targetType" TEXT,
    "suspendedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "GitHubInstallation_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "UserGitHubInstallation" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "githubInstallationId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "UserGitHubInstallation_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "GitHubOAuthState" (
    "id" TEXT NOT NULL,
    "stateHash" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "githubInstallationId" TEXT,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "usedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "GitHubOAuthState_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "Repository" (
    "id" TEXT NOT NULL,
    "githubRepositoryId" TEXT NOT NULL,
    "githubInstallationId" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "fullName" TEXT NOT NULL,
    "visibility" "RepositoryVisibility" NOT NULL,
    "defaultBranch" TEXT NOT NULL,
    "archived" BOOLEAN NOT NULL DEFAULT false,
    "htmlUrl" TEXT,
    "accessStatus" "RepositoryAccessStatus" NOT NULL DEFAULT 'ACTIVE',
    "indexStatus" "RepositoryIndexStatus" NOT NULL DEFAULT 'NOT_INDEXED',
    "indexProgress" INTEGER NOT NULL DEFAULT 0,
    "indexedCommitSha" TEXT,
    "targetCommitSha" TEXT,
    "indexedFileCount" INTEGER NOT NULL DEFAULT 0,
    "indexedChunkCount" INTEGER NOT NULL DEFAULT 0,
    "lastIndexedAt" TIMESTAMP(3),
    "lastIndexError" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "Repository_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "UserRepository" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "repositoryId" TEXT NOT NULL,
    "accessStatus" "RepositoryAccessStatus" NOT NULL DEFAULT 'ACTIVE',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "UserRepository_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "RepositoryIndexJob" (
    "id" TEXT NOT NULL,
    "repositoryId" TEXT NOT NULL,
    "trigger" "RepositoryIndexJobTrigger" NOT NULL,
    "status" "RepositoryIndexJobStatus" NOT NULL DEFAULT 'QUEUED',
    "sourceCommitSha" TEXT,
    "targetCommitSha" TEXT,
    "progress" INTEGER NOT NULL DEFAULT 0,
    "errorMessage" TEXT,
    "startedAt" TIMESTAMP(3),
    "finishedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "RepositoryIndexJob_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "RepositoryFile" (
    "id" TEXT NOT NULL,
    "repositoryId" TEXT NOT NULL,
    "path" TEXT NOT NULL,
    "blobSha" TEXT NOT NULL,
    "contentSha" TEXT NOT NULL,
    "byteSize" INTEGER NOT NULL,
    "lineCount" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "RepositoryFile_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "RepositoryChunk" (
    "id" TEXT NOT NULL,
    "repositoryId" TEXT NOT NULL,
    "fileId" TEXT NOT NULL,
    "chunkIndex" INTEGER NOT NULL,
    "path" TEXT NOT NULL,
    "startLine" INTEGER NOT NULL,
    "endLine" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "contentSha" TEXT NOT NULL,
    "embedding" vector(1536) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "RepositoryChunk_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "GitHubInstallation_githubInstallationId_key" ON "GitHubInstallation"("githubInstallationId");
CREATE INDEX "GitHubInstallation_accountLogin_idx" ON "GitHubInstallation"("accountLogin");
CREATE UNIQUE INDEX "UserGitHubInstallation_userId_githubInstallationId_key" ON "UserGitHubInstallation"("userId", "githubInstallationId");
CREATE INDEX "UserGitHubInstallation_githubInstallationId_idx" ON "UserGitHubInstallation"("githubInstallationId");
CREATE UNIQUE INDEX "GitHubOAuthState_stateHash_key" ON "GitHubOAuthState"("stateHash");
CREATE INDEX "GitHubOAuthState_userId_idx" ON "GitHubOAuthState"("userId");
CREATE INDEX "GitHubOAuthState_expiresAt_idx" ON "GitHubOAuthState"("expiresAt");
CREATE UNIQUE INDEX "Repository_githubRepositoryId_key" ON "Repository"("githubRepositoryId");
CREATE INDEX "Repository_githubInstallationId_idx" ON "Repository"("githubInstallationId");
CREATE INDEX "Repository_owner_name_idx" ON "Repository"("owner", "name");
CREATE INDEX "Repository_indexStatus_idx" ON "Repository"("indexStatus");
CREATE INDEX "Repository_accessStatus_idx" ON "Repository"("accessStatus");
CREATE UNIQUE INDEX "UserRepository_userId_repositoryId_key" ON "UserRepository"("userId", "repositoryId");
CREATE INDEX "UserRepository_repositoryId_idx" ON "UserRepository"("repositoryId");
CREATE INDEX "UserRepository_accessStatus_idx" ON "UserRepository"("accessStatus");
CREATE INDEX "RepositoryIndexJob_repositoryId_status_idx" ON "RepositoryIndexJob"("repositoryId", "status");
CREATE INDEX "RepositoryIndexJob_createdAt_idx" ON "RepositoryIndexJob"("createdAt");
CREATE UNIQUE INDEX "RepositoryIndexJob_one_active_per_repository_idx" ON "RepositoryIndexJob"("repositoryId") WHERE "status" IN ('QUEUED', 'RUNNING');
CREATE UNIQUE INDEX "RepositoryFile_repositoryId_path_key" ON "RepositoryFile"("repositoryId", "path");
CREATE INDEX "RepositoryFile_repositoryId_blobSha_idx" ON "RepositoryFile"("repositoryId", "blobSha");
CREATE UNIQUE INDEX "RepositoryChunk_fileId_chunkIndex_key" ON "RepositoryChunk"("fileId", "chunkIndex");
CREATE INDEX "RepositoryChunk_repositoryId_path_idx" ON "RepositoryChunk"("repositoryId", "path");
CREATE INDEX "RepositoryChunk_embedding_hnsw_idx" ON "RepositoryChunk" USING hnsw ("embedding" vector_cosine_ops);

ALTER TABLE "UserGitHubInstallation" ADD CONSTRAINT "UserGitHubInstallation_githubInstallationId_fkey" FOREIGN KEY ("githubInstallationId") REFERENCES "GitHubInstallation"("githubInstallationId") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "UserGitHubInstallation" ADD CONSTRAINT "UserGitHubInstallation_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "Repository" ADD CONSTRAINT "Repository_githubInstallationId_fkey" FOREIGN KEY ("githubInstallationId") REFERENCES "GitHubInstallation"("githubInstallationId") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "UserRepository" ADD CONSTRAINT "UserRepository_repositoryId_fkey" FOREIGN KEY ("repositoryId") REFERENCES "Repository"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "UserRepository" ADD CONSTRAINT "UserRepository_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "RepositoryIndexJob" ADD CONSTRAINT "RepositoryIndexJob_repositoryId_fkey" FOREIGN KEY ("repositoryId") REFERENCES "Repository"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "RepositoryFile" ADD CONSTRAINT "RepositoryFile_repositoryId_fkey" FOREIGN KEY ("repositoryId") REFERENCES "Repository"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "RepositoryChunk" ADD CONSTRAINT "RepositoryChunk_fileId_fkey" FOREIGN KEY ("fileId") REFERENCES "RepositoryFile"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "RepositoryChunk" ADD CONSTRAINT "RepositoryChunk_repositoryId_fkey" FOREIGN KEY ("repositoryId") REFERENCES "Repository"("id") ON DELETE CASCADE ON UPDATE CASCADE;
