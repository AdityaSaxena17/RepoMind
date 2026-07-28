import { UserButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import {
  Bot,
  Code2,
  GitPullRequest,
  LayoutDashboard,
  Search,
  Settings,
  Sparkles,
  TerminalSquare,
} from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import type { RepositoryView } from "../lib/repository-view";

type ActiveItem =
  | "dashboard"
  | "repositories"
  | "chat"
  | "search"
  | "issues"
  | "pull-requests"
  | "settings";

function navigationHref(activeItem: ActiveItem, repository: RepositoryView) {
  const base = `/repositories/${repository.owner}/${repository.name}`;

  if (activeItem === "search") {
    return `${base}/search`;
  }

  if (activeItem === "issues") {
    return `${base}/issues`;
  }

  if (activeItem === "pull-requests") {
    return `${base}/pull-requests`;
  }

  if (activeItem === "chat") {
    return base;
  }

  return "/dashboard";
}

const globalNavigationItems: {
  id: Extract<ActiveItem, "dashboard" | "repositories" | "settings">;
  label: string;
  href: string;
  icon: typeof LayoutDashboard;
}[] = [
  { id: "dashboard", label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  {
    id: "repositories",
    label: "Repositories",
    href: "/dashboard",
    icon: Code2,
  },
  { id: "settings", label: "Settings", href: "/settings", icon: Settings },
];

const repositoryNavigationItems: {
  id: ActiveItem;
  label: string;
  icon: typeof LayoutDashboard;
}[] = [
  { id: "chat", label: "AI Chat", icon: Bot },
  { id: "search", label: "Search", icon: Search },
  { id: "issues", label: "Issues", icon: TerminalSquare },
  { id: "pull-requests", label: "Pull Requests", icon: GitPullRequest },
];

export async function AppShell({
  activeItem,
  repository,
  rightPanel,
  children,
}: {
  activeItem: ActiveItem;
  repository?: RepositoryView;
  rightPanel?: ReactNode;
  children: ReactNode;
}) {
  await auth.protect();

  const selectedRepositoryLabel = repository
    ? `${repository.owner}/${repository.name}`
    : "No repository selected";
  const shellGridClass = rightPanel
    ? "grid min-h-[calc(100svh-4rem)] grid-cols-1 md:grid-cols-[232px_minmax(0,1fr)] xl:grid-cols-[232px_minmax(0,1fr)_336px]"
    : "grid min-h-[calc(100svh-4rem)] grid-cols-1 md:grid-cols-[232px_minmax(0,1fr)]";

  return (
    <div className="min-h-svh bg-[#09090b] text-zinc-100">
      <header className="sticky top-0 z-20 flex h-16 items-center border-b border-white/10 bg-[#09090b]/95 px-4 backdrop-blur md:px-6">
        <Link
          className="flex min-w-44 items-center gap-2 text-sm font-semibold"
          href="/dashboard"
          prefetch={false}
        >
          <span className="grid h-8 w-8 place-items-center rounded-md border border-sky-400/30 bg-sky-400/10 text-sky-200">
            <Sparkles className="h-4 w-4" />
          </span>
          RepoMind
        </Link>
        <div className="hidden flex-1 items-center gap-3 md:flex">
          <div className="flex h-9 min-w-56 items-center rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-300">
            {selectedRepositoryLabel}
          </div>
          {repository ? (
            <div className="flex h-9 flex-1 max-w-xl items-center gap-2 rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-500">
              <Search className="h-4 w-4" />
              <span>Search code, issues, pull requests...</span>
            </div>
          ) : null}
        </div>
        <div className="ml-auto flex items-center gap-3">
          <UserButton />
        </div>
      </header>
      <div className={shellGridClass}>
        <aside className="hidden border-r border-white/10 bg-[#0c0c0f] p-3 md:block">
          <nav className="space-y-1">
            {globalNavigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeItem === item.id;

              return (
                <Link
                  className={`flex h-9 items-center gap-3 rounded-md px-3 text-sm transition ${
                    isActive
                      ? "bg-white/10 text-zinc-50"
                      : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-100"
                  }`}
                  href={item.href}
                  key={item.id}
                  prefetch={false}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
            {repository ? (
              <>
                <div className="px-3 pb-1 pt-4 text-xs font-medium uppercase tracking-wide text-zinc-600">
                  Repository
                </div>
                {repositoryNavigationItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeItem === item.id;

                  return (
                    <Link
                      className={`flex h-9 items-center gap-3 rounded-md px-3 text-sm transition ${
                        isActive
                          ? "bg-white/10 text-zinc-50"
                          : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-100"
                      }`}
                      href={navigationHref(item.id, repository)}
                      key={item.id}
                      prefetch={false}
                    >
                      <Icon className="h-4 w-4" />
                      {item.label}
                    </Link>
                  );
                })}
              </>
            ) : null}
          </nav>
          <div className="mt-8 rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <p className="text-xs font-medium text-zinc-300">Indexing flow</p>
            <p className="mt-2 text-xs leading-5 text-zinc-500">
              Repository connection requests go through Next.js, then FastAPI,
              then arq Redis.
            </p>
          </div>
        </aside>
        <main className="min-w-0">{children}</main>
        {rightPanel ? (
          <aside className="hidden border-l border-white/10 bg-[#0c0c0f] p-5 xl:block">
            {rightPanel}
          </aside>
        ) : null}
      </div>
    </div>
  );
}
