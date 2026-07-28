import {
  ArrowRight,
  GitBranch,
  Lock,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import Link from "next/link";
import type { Repository } from "../lib/prototype-data";
import { StatusBadge } from "./status-badge";

export function RepositoryCard({ repository }: { repository: Repository }) {
  const Icon = repository.visibility === "Private" ? Lock : ShieldCheck;

  return (
    <Link
      className="group flex min-h-64 flex-col justify-between rounded-lg border border-white/10 bg-zinc-950/70 p-5 transition hover:border-white/20 hover:bg-zinc-950"
      href={`/repositories/${repository.owner}/${repository.name}`}
      prefetch={false}
    >
      <div>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="flex items-center gap-2 text-xs text-zinc-500">
              <Icon className="h-3.5 w-3.5" />
              {repository.owner}
            </p>
            <h2 className="mt-2 truncate text-lg font-semibold text-zinc-50">
              {repository.name}
            </h2>
          </div>
          <StatusBadge status={repository.status} />
        </div>
        <p className="mt-4 line-clamp-2 text-sm leading-6 text-zinc-400">
          {repository.description}
        </p>
        <div className="mt-5 flex flex-wrap gap-2 text-xs text-zinc-400">
          <span className="rounded-md border border-white/10 px-2 py-1">
            {repository.language}
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1">
            <GitBranch className="h-3 w-3" />
            {repository.branch}
          </span>
        </div>
      </div>
      <div className="mt-6">
        {repository.status === "INDEXING" ||
        repository.status === "UPDATING" ||
        repository.status === "FAILED" ? (
          <div>
            <div className="mb-2 flex items-center justify-between text-xs text-zinc-500">
              <span className="inline-flex items-center gap-1">
                <RefreshCw className="h-3 w-3" />
                Index progress
              </span>
              <span>{repository.progress}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-sky-400"
                style={{ width: `${repository.progress}%` }}
              />
            </div>
          </div>
        ) : null}
        <div className="mt-5 flex items-center justify-between text-xs text-zinc-500">
          <span>Last indexed {repository.lastIndexed}</span>
          <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5 group-hover:text-zinc-200" />
        </div>
      </div>
    </Link>
  );
}
