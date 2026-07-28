# RepoMind Database Schema

RepoMind stores public GitHub repositories by `owner/repo`, tracks which Clerk
users connected them, and runs indexing jobs asynchronously through Redis/arq.

## Core Tables

- `User`: Clerk-synced application user.
- `Repository`: public GitHub repository metadata, index status, current commit
  pointers, file count, chunk count, and last indexing error.
- `UserRepository`: join table connecting RepoMind users to connected
  repositories.
- `RepositoryIndexJob`: durable record for each indexing request, including
  trigger, status, progress, commit range, error message, and timestamps.
- `RepositoryFile`: indexed source files with path, content hash, byte size,
  line count, and content.
- `RepositoryChunk`: chunk metadata and `vector(1536)` embeddings for pgvector.

## Current Boundaries

- Only public GitHub repositories are supported.
- Repositories are connected by entering `owner` and `repo`.
- Webhooks, GitHub account linking, private repositories, and automatic push
  updates are intentionally out of scope for this version.
- Retrieval strategy is deferred; indexing stores data for the later retriever.
