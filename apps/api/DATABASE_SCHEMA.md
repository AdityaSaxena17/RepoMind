# RepoMind Database Schema Plan

The first database-backed implementation should keep GitHub installation state,
repository access, and indexing jobs separate. This lets the UI show connected
repositories immediately while indexing work proceeds asynchronously through
Redis/arq.

## Core Tables

- `User`: existing Clerk-synced application user.
- `GitHubInstallation`: GitHub App installation id, account login/type, target
  type, timestamps, and removal timestamp.
- `Repository`: GitHub repository id, owner, name, full name, visibility,
  primary language, default branch, HTML URL, current index status, last indexed
  timestamp, and installation relation.
- `UserRepository`: join table connecting RepoMind users to repositories they
  can access through an installation.
- `RepositoryIndexJob`: durable record for each indexing request, including the
  arq job id, trigger, status, progress, error message, start/end timestamps,
  and repository relation.

## Future Indexing Tables

- `RepositoryFile`: one row per indexed file with path, language, content hash,
  byte size, and last indexed timestamp.
- `RepositoryChunk`: chunk metadata for RAG, including file relation, line range,
  token count, content hash, and embedding id.
- `RepositoryEmbedding`: vector storage row once pgvector is enabled. In Prisma,
  the vector column should use `Unsupported("vector")` unless the project adopts
  a first-class vector mapping before implementation.

## Status Values

- Repository index status: `READY`, `INDEXING`, `UPDATING`, `FAILED`,
  `NOT_INDEXED`.
- Index job status: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.
