export type RepositoryStatus =
  | "READY"
  | "INDEXING"
  | "UPDATING"
  | "FAILED"
  | "NOT_INDEXED";

export type TimelineStatus = "complete" | "active" | "pending";

export interface Repository {
  owner: string;
  name: string;
  visibility: "Public" | "Private";
  language: string;
  description: string;
  branch: string;
  status: RepositoryStatus;
  lastIndexed: string;
  progress: number;
  indexedFiles: number;
  embeddings: number;
  openIssues: number;
  pullRequests: number;
}

export interface Citation {
  filePath: string;
  startLine: number;
  endLine: number;
  label: string;
}

export interface TimelineStep {
  label: string;
  status: TimelineStatus;
  detail?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: string;
  messages: ChatMessage[];
  followUps: string[];
  timeline: TimelineStep[];
}

export interface SearchResult {
  filePath: string;
  symbolName: string;
  explanation: string;
  confidence: number;
  citations: Citation[];
}

export interface IssuePlan {
  issueNumber: number;
  title: string;
  status: "Open" | "In progress" | "Triaged";
  summary: string;
  likelyFiles: string[];
  relatedImplementation: string;
  implementationSteps: string[];
  edgeCases: string[];
  testingNotes: string[];
}

export interface PullRequestContext {
  number: number;
  title: string;
  author: string;
  status: "Draft" | "Open" | "Review requested";
  changedFiles: number;
  summary: string;
  timeline: TimelineStep[];
}

export const repositories: Repository[] = [
  {
    owner: "vercel",
    name: "commerce",
    visibility: "Public",
    language: "TypeScript",
    description: "Composable storefront with product, cart, and checkout flows.",
    branch: "main",
    status: "READY",
    lastIndexed: "12 minutes ago",
    progress: 100,
    indexedFiles: 1284,
    embeddings: 18492,
    openIssues: 37,
    pullRequests: 12,
  },
  {
    owner: "acme",
    name: "payments-api",
    visibility: "Private",
    language: "Go",
    description: "Internal payment orchestration service for billing events.",
    branch: "release/2026-07",
    status: "INDEXING",
    lastIndexed: "Indexing now",
    progress: 64,
    indexedFiles: 746,
    embeddings: 8110,
    openIssues: 9,
    pullRequests: 4,
  },
  {
    owner: "open-source-labs",
    name: "queuekit",
    visibility: "Public",
    language: "Rust",
    description: "Durable job runner with retry and scheduler primitives.",
    branch: "main",
    status: "UPDATING",
    lastIndexed: "2 hours ago",
    progress: 82,
    indexedFiles: 512,
    embeddings: 6218,
    openIssues: 18,
    pullRequests: 7,
  },
  {
    owner: "northstar",
    name: "mobile-backend",
    visibility: "Private",
    language: "Kotlin",
    description: "API surface for mobile onboarding, sessions, and feature flags.",
    branch: "trunk",
    status: "FAILED",
    lastIndexed: "Yesterday",
    progress: 41,
    indexedFiles: 302,
    embeddings: 3904,
    openIssues: 21,
    pullRequests: 3,
  },
];

export const conversations: Conversation[] = [
  {
    id: "auth-flow",
    title: "Explain the authentication flow",
    updatedAt: "Just now",
    messages: [
      {
        role: "user",
        content: "Explain the authentication flow and where sessions are created.",
      },
      {
        role: "assistant",
        content:
          "Authentication enters through the Next.js middleware, which gates protected routes before rendering. The session is then resolved in server components and API handlers, while user records are synchronized from Clerk webhooks into Postgres. The important boundary is that UI routes trust Clerk session state, and the database user row is used for RepoMind-owned metadata rather than primary authentication.",
        citations: [
          {
            filePath: "apps/web/proxy.ts",
            startLine: 1,
            endLine: 11,
            label: "Clerk middleware",
          },
          {
            filePath: "apps/web/app/api/webhooks/clerk/route.ts",
            startLine: 9,
            endLine: 65,
            label: "User sync webhook",
          },
          {
            filePath: "packages/db/prisma/schema.prisma",
            startLine: 15,
            endLine: 31,
            label: "User model",
          },
        ],
      },
    ],
    followUps: [
      "Where should repository access checks live?",
      "Show the webhook failure modes.",
      "Plan GitHub App installation storage.",
    ],
    timeline: [
      {
        label: "Understanding the question",
        status: "complete",
        detail: "Detected auth/session and user persistence intent.",
      },
      {
        label: "Searching repository",
        status: "complete",
        detail: "Matched Clerk middleware, dashboard access, and webhook route.",
      },
      {
        label: "Verifying references",
        status: "complete",
        detail: "Checked exact source files and schema fields.",
      },
      {
        label: "Generating response",
        status: "active",
        detail: "Grounded summary with citations.",
      },
    ],
  },
];

export const searchResults: SearchResult[] = [
  {
    filePath: "apps/web/app/api/webhooks/clerk/route.ts",
    symbolName: "POST",
    explanation:
      "Handles Clerk user lifecycle webhooks and upserts or deactivates local users.",
    confidence: 96,
    citations: [
      {
        filePath: "apps/web/app/api/webhooks/clerk/route.ts",
        startLine: 7,
        endLine: 70,
        label: "Webhook handler",
      },
    ],
  },
  {
    filePath: "apps/web/app/dashboard/page.tsx",
    symbolName: "Dashboard",
    explanation:
      "Protects the dashboard with Clerk and reads the synced user record from Prisma.",
    confidence: 91,
    citations: [
      {
        filePath: "apps/web/app/dashboard/page.tsx",
        startLine: 4,
        endLine: 37,
        label: "Protected dashboard",
      },
    ],
  },
  {
    filePath: "packages/db/src/client.ts",
    symbolName: "prisma",
    explanation:
      "Creates the shared Prisma client and reuses it during local development.",
    confidence: 88,
    citations: [
      {
        filePath: "packages/db/src/client.ts",
        startLine: 1,
        endLine: 11,
        label: "Prisma singleton",
      },
    ],
  },
];

export const issuePlans: IssuePlan[] = [
  {
    issueNumber: 42,
    title: "Add repository installation synchronization",
    status: "Triaged",
    summary:
      "Persist GitHub App installations and expose accessible repositories on the dashboard.",
    likelyFiles: [
      "packages/db/prisma/schema.prisma",
      "apps/web/app/api/webhooks/github/route.ts",
      "apps/web/app/dashboard/page.tsx",
    ],
    relatedImplementation:
      "The Clerk webhook already shows the preferred pattern for verified webhook handling and idempotent upserts.",
    implementationSteps: [
      "Add installation and repository models scoped to the synced user.",
      "Verify GitHub webhook signatures before processing installation events.",
      "Upsert repository metadata and mark repositories as not indexed initially.",
      "Render connected repositories from Prisma instead of prototype data.",
    ],
    edgeCases: [
      "Installation removed while an index is running.",
      "Repository transferred to a new owner.",
      "Private repository access revoked before refresh.",
    ],
    testingNotes: [
      "Regression test installation create/update/delete events.",
      "Verify dashboard hides repositories no longer accessible to the user.",
    ],
  },
  {
    issueNumber: 57,
    title: "Prototype natural language code search",
    status: "Open",
    summary:
      "Create the first searchable repository page before wiring embeddings.",
    likelyFiles: [
      "apps/web/app/repositories/[owner]/[repo]/search/page.tsx",
      "apps/web/lib/prototype-data.ts",
    ],
    relatedImplementation:
      "Static search results can mirror the eventual file, symbol, explanation, and confidence response shape.",
    implementationSteps: [
      "Add a search page with a prominent query input.",
      "Render file references and confidence indicators.",
      "Reserve empty/loading states for the real retrieval API.",
    ],
    edgeCases: [
      "No matching repository selected.",
      "Search query is too vague to rank confidently.",
    ],
    testingNotes: [
      "Check responsive result rows on narrow screens.",
      "Typecheck citation and confidence fields.",
    ],
  },
];

export const pullRequests: PullRequestContext[] = [
  {
    number: 18,
    title: "Wire Clerk webhook into local user store",
    author: "aditya",
    status: "Review requested",
    changedFiles: 5,
    summary:
      "Adds verified Clerk webhook handling, Prisma user persistence, and dashboard sync status.",
    timeline: [
      {
        label: "Fetched PR metadata",
        status: "complete",
      },
      {
        label: "Compared changed files",
        status: "complete",
      },
      {
        label: "Mapped behavior to auth flow",
        status: "active",
      },
    ],
  },
  {
    number: 24,
    title: "Add GitHub App environment placeholders",
    author: "repo-bot",
    status: "Open",
    changedFiles: 2,
    summary:
      "Documents the GitHub App credentials the ingestion layer will need later.",
    timeline: [
      {
        label: "Read environment template",
        status: "complete",
      },
      {
        label: "Checked affected configuration",
        status: "active",
      },
      {
        label: "Generate summary",
        status: "pending",
      },
    ],
  },
];

export function getRepository(owner: string, name: string) {
  return repositories.find(
    (repository) => repository.owner === owner && repository.name === name,
  );
}
