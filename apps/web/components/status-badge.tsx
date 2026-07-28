import type { RepositoryStatus } from "../lib/prototype-data";

const statusStyles: Record<RepositoryStatus, string> = {
  READY: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
  INDEXING: "border-sky-400/30 bg-sky-400/10 text-sky-200",
  UPDATING: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  FAILED: "border-rose-400/30 bg-rose-400/10 text-rose-200",
  NOT_INDEXED: "border-zinc-500/40 bg-zinc-500/10 text-zinc-300",
};

const statusLabels: Record<RepositoryStatus, string> = {
  READY: "Ready",
  INDEXING: "Indexing",
  UPDATING: "Updating",
  FAILED: "Failed",
  NOT_INDEXED: "Not indexed",
};

export function StatusBadge({ status }: { status: RepositoryStatus }) {
  return (
    <span
      className={`inline-flex h-6 items-center rounded-full border px-2.5 text-xs font-medium ${statusStyles[status]}`}
    >
      {statusLabels[status]}
    </span>
  );
}
