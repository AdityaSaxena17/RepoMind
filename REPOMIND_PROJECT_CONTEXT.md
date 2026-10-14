# RepoMind Project Context

**GitHub repository:** https://github.com/AdityaSaxena17/RepoMind

## Resume-Ready Summary

RepoMind is a full-stack developer tool that lets users provide a public GitHub repository link, index its source code, and ask repository-grounded questions in an AI chat workspace. No GitHub OAuth or GitHub App installation is required for repository access: RepoMind validates the owner/repository input and reads public GitHub data through the GitHub REST API. The application combines a Next.js web client, a FastAPI service, PostgreSQL with pgvector, Redis-backed background jobs, OpenAI embeddings/chat models, and a repository-scoped MCP server for live GitHub issues and pull requests.

## One-Line Description

A citation-aware AI code exploration platform for indexing public GitHub repositories and answering questions about code, issues, and pull requests from a direct repository link.

## Resume Bullet Options

- Built RepoMind, a full-stack AI developer tool that indexes public GitHub repositories and provides citation-backed conversational code exploration.
- Implemented an asynchronous repository ingestion pipeline with FastAPI, ARQ/Redis, GitHub REST APIs, OpenAI embeddings, PostgreSQL, and pgvector.
- Designed hybrid retrieval that fuses semantic vector search with PostgreSQL full-text search, expands matching chunks into bounded source windows, and returns line-addressable citations.
- Integrated LangChain agents with a repository-scoped Model Context Protocol server to answer live GitHub issue and pull-request questions without crossing repository access boundaries.
- Developed a Next.js/React repository workspace with direct public-repository input, repository connection and retry flows, indexing status polling, streaming chat, tool activity indicators, conversation history, and source links.
- Added transactional index promotion and explicit indexing/chat status models so failed indexing jobs do not silently appear as usable repository context.

## Problem and Product Value

Large repositories are difficult to understand through ordinary file browsing and keyword search. RepoMind creates a searchable representation of a connected repository and gives developers a conversational interface for questions such as:

- Where is a feature or API implemented?
- How do modules or services interact?
- What does a particular file or code path do?
- Which open issues or pull requests relate to a topic?
- What source lines support the answer?

The product distinguishes between indexed source-code knowledge and live GitHub collaboration data. Source-code claims come from the indexed commit, while issue and pull-request claims use live, repository-scoped GitHub MCP tools.

## Architecture

```text
Next.js web app with direct public GitHub repository input
        |
        | same-origin API routes validate repository input and requests
        v
Next.js server routes + Prisma client
        |                         \
        | enqueue indexing          \ stream chat request
        v                            v
FastAPI service -------------- PostgreSQL / pgvector
        |
        v
Redis + ARQ worker ---- GitHub REST API
        |
        v
OpenAI embeddings -> RepositoryChunk.vector(1536)

Chat path:
PostgreSQL retrieval + conversation history
        +
repository-scoped MCP stdio server -> live GitHub issues/PRs
        |
        v
LangChain agent + OpenAI chat model -> SSE events -> React chat UI
```

### Application Layers

#### Web application: `apps/web`

- Next.js 16 application using React 19 and TypeScript.
- Repository access uses direct public repository input rather than GitHub OAuth or a GitHub App installation.
- Server-side route handlers validate repository, chat, GitHub, and webhook requests.
- Prisma-backed helpers store connected public repository metadata and indexing state.
- Repository dashboard supports connection, filtering, status refresh, retry indexing, and navigation into a repository workspace.
- Repository workspace includes chat, issue, and pull-request views.
- React chat client consumes `text/event-stream` responses and renders assistant deltas, tool activity, citations, errors, and persisted conversation history.
- Shared UI, ESLint, Tailwind, and TypeScript configuration are maintained as workspace packages.

#### API service: `apps/api`

- FastAPI application exposes versioned endpoints under `/api/v1`.
- `/healthz` reports service health and the configured chat model state.
- Repository connection requests enqueue idempotent ARQ jobs for background indexing.
- Chat requests validate the internal credential, open a database connection, and stream server-sent events.
- Configuration is centralized and validated through the API settings module.
- Strict mypy, Ruff, and pytest configuration is defined in `pyproject.toml`.

#### Data layer: `packages/db`

- Prisma schema targets PostgreSQL and is shared by the TypeScript application.
- PostgreSQL uses the pgvector extension for 1,536-dimensional embeddings.
- Repository files preserve source content, GitHub blob SHA, content SHA, byte size, and line count.
- Repository chunks preserve source path, line range, content hash, and vector embedding.
- Chat conversations, messages, message statuses, and source citations are persisted.
- Access and lifecycle enums model active/revoked relationships, indexing states, job states, and message states.

#### Local infrastructure

- PostgreSQL runs from `pgvector/pgvector:0.8.2-pg17`.
- Redis provides the ARQ queue and persistence for local queue inspection.
- RedisInsight and pgAdmin are included for local operations and debugging.
- An ARQ dashboard container exposes indexing queue visibility.
- Turborepo and pnpm coordinate JavaScript/TypeScript applications and packages.

## Core Workflows

### 1. Connect and index a repository

1. A user submits a public GitHub owner and repository name in the dashboard; no GitHub OAuth flow is required.
2. The Next.js route validates the request, fetches repository metadata, and upserts the repository and user-repository relationship.
3. The web layer creates an index-job record and calls the FastAPI enqueue endpoint.
4. FastAPI enqueues an ARQ job using a deterministic repository/job key so repeated requests can be recognized as already queued.
5. The worker resolves the repository’s default-branch head commit and fetches the Git tree recursively.
6. Oversized files, generated/build directories, dependency directories, lockfiles, maps, minified assets, environment files, and private-key-like content are excluded.
7. Remaining UTF-8 files are stored and split with a tiktoken-based recursive splitter using 800-token chunks and 100-token overlap.
8. OpenAI embeddings are generated asynchronously in batches.
9. Files and chunks are promoted inside a database transaction. The repository is marked `READY` only after the complete replacement index is written.
10. Failures update both repository and job error state with redacted messages, allowing the UI to offer retry indexing.

### 2. Ask a repository question

1. The repository page verifies that the user has an active relationship with the repository.
2. The API verifies that indexing is ready and that the submitted message belongs to the user’s repository conversation.
3. The retriever runs vector similarity search and PostgreSQL full-text search concurrently.
4. Reciprocal Rank Fusion combines the candidate lists, deduplicates chunks, and selects the strongest results.
5. Matching child chunks expand into bounded parent file windows, merging nearby windows and limiting context to six final source windows, 250 lines, and 16,000 characters per window.
6. The agent receives recent conversation history and formatted source context with numbered references.
7. The LangChain agent can use the scoped GitHub MCP tools for live issue and pull-request questions.
8. Assistant text, tool activity, citations, and completion status are streamed as SSE events.
9. The final assistant response and source-line citations are persisted for future conversation loading.

### 3. Read live GitHub collaboration data

The FastMCP server exposes repository-scoped tools for searching and reading issues and pull requests, including comments. Before any GitHub request, the server resolves the user and repository IDs from the process environment and verifies an active `UserRepository` relationship in PostgreSQL. It then applies the repository constraint to GitHub search queries, validates pagination bounds, normalizes response data, and redacts user-facing HTTP errors.

## Technical Highlights

### Retrieval and grounding

- Semantic retrieval uses OpenAI embeddings and pgvector cosine distance.
- Keyword retrieval uses PostgreSQL `websearch_to_tsquery` and `ts_rank_cd`.
- Reciprocal Rank Fusion combines semantic and lexical evidence using a configurable rank constant.
- Parent-window expansion improves context continuity around matching chunks.
- Every retrieved source retains path and line metadata.
- The chat system prompt directs the agent to cite code with source numbers, cite GitHub data with returned URLs, and state when evidence is insufficient.

### Reliability and consistency

- Background indexing keeps GitHub traversal and embedding work out of request handlers.
- Deterministic ARQ job IDs reduce duplicate enqueue behavior.
- Index replacement and repository readiness updates occur in one transaction.
- Explicit `NOT_INDEXED`, `INDEXING`, `UPDATING`, `READY`, and `FAILED` states are surfaced to the UI.
- Chat messages distinguish pending, streaming, succeeded, and failed states.
- SSE errors are persisted and sent to the client instead of leaving an indefinite streaming state.
- GitHub rate-limit, authorization, and missing-resource cases receive controlled errors.

### Security and access control

- GitHub repository access is limited to public repositories supplied directly by owner and name.
- Repository pages and server routes validate the requested repository before starting work.
- The MCP server rechecks repository scope against PostgreSQL rather than trusting only client input.
- Internal API credentials protect the FastAPI enqueue and chat endpoints when configured.
- Repository scope is enforced in SQL and in GitHub search query construction.
- Potential secrets and private-key content are excluded from indexing.
- Error messages are truncated and normalized before persistence or streaming.

## Repository Structure

```text
apps/api/                   FastAPI service, worker, retrieval, MCP server, and tests
apps/web/                   Next.js application, route handlers, pages, and React components
packages/db/                Prisma schema, migrations, and generated database client
packages/ui/                Shared UI components and styles
packages/eslint-config/     Shared linting presets
packages/tailwind-config/   Shared Tailwind/PostCSS configuration
packages/typescript-config/ Shared TypeScript compiler configurations
docker/                     Local pgAdmin configuration
docker-compose.yml          PostgreSQL, Redis, RedisInsight, pgAdmin, and ARQ dashboard
```

## Testing and Quality Tooling

### Python API

- `pytest` test suite covers chat behavior, configuration, GitHub MCP serialization/scoping, repository routes, retrieval, and worker GitHub ingestion behavior.
- `mypy` runs in strict mode for Python 3.10.
- `ruff` covers errors, imports, modern Python syntax, bugbear checks, and simplification rules.

### Web and monorepo

- Turborepo coordinates build, development, lint, test, and type-check tasks.
- ESLint runs with zero warnings for the web app.
- Next.js type generation and TypeScript checking are configured for the web application.
- Prettier formats TypeScript, TSX, and Markdown files.

## Useful Development Commands

```sh
# JavaScript/TypeScript workspace
pnpm install
pnpm dev
pnpm build
pnpm lint
pnpm check-types

# Local infrastructure
docker compose up -d

# Python API environment and tests
cd apps/api
uv sync
uv run pytest
uv run mypy app tests
uv run ruff check app tests
```

## Resume Talking Points

When discussing this project, emphasize the engineering decisions rather than only listing frameworks:

- Retrieval quality: hybrid semantic plus lexical search, rank fusion, and parent-context expansion.
- Trustworthiness: commit-pinned indexing, source-line citations, evidence-aware prompting, and explicit uncertainty behavior.
- Scale-minded processing: asynchronous jobs, bounded file/chunk/context sizes, batching, and transactional index promotion.
- Secure tool use: MCP tools are scoped by a database-backed user-repository authorization check.
- Product completeness: direct public-repository links, repository lifecycle states, retry behavior, streaming UX, persisted conversations, and live collaboration data.

## Claims to Quantify Before Finalizing a Resume

The codebase verifies the capabilities above but does not establish production traffic or performance numbers. Before adding metrics, measure and record:

- Number of repositories or files indexed in a representative run.
- Indexing duration and embedding throughput.
- Retrieval latency and chat time-to-first-token.
- Test count and CI pass rate.
- Deployment scale, user count, or query volume if applicable.

Avoid claiming unsupported production scale, accuracy, latency, or cost improvements.
