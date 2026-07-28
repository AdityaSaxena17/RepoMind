"use client";

import { CheckCircle2, Loader2, Send, TriangleAlert } from "lucide-react";
import { useState } from "react";

interface ConnectRepositoryResponse {
  job_id: string;
  enqueued: boolean;
  queue_name: string;
  repository: string;
}

function readOptionalPositiveInteger(value: FormDataEntryValue | null) {
  if (typeof value !== "string" || value.trim().length === 0) {
    return undefined;
  }

  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

export function ConnectRepositoryButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);

  async function connectRepository(formData: FormData) {
    const owner = formData.get("owner");
    const repo = formData.get("repo");
    const defaultBranch = formData.get("default_branch");

    if (
      typeof owner !== "string" ||
      owner.trim().length === 0 ||
      typeof repo !== "string" ||
      repo.trim().length === 0
    ) {
      setStatus("error");
      setMessage("Owner and repository name are required.");
      return;
    }

    setStatus("loading");
    setMessage(null);

    const response = await fetch("/api/repositories/connect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        owner: owner.trim(),
        repo: repo.trim(),
        github_repository_id: readOptionalPositiveInteger(
          formData.get("github_repository_id"),
        ),
        installation_id: readOptionalPositiveInteger(formData.get("installation_id")),
        default_branch:
          typeof defaultBranch === "string" && defaultBranch.trim().length > 0
            ? defaultBranch.trim()
            : undefined,
        private: formData.get("private") === "on",
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      setStatus("error");
      setMessage(text || "FastAPI rejected the repository connection.");
      return;
    }

    const result = (await response.json()) as ConnectRepositoryResponse;
    setStatus("success");
    setMessage(
      result.enqueued
        ? `Queued ${result.repository} on ${result.queue_name}. Job: ${result.job_id}`
        : `${result.repository} already has a queued job: ${result.job_id}`,
    );
  }

  return (
    <form action={connectRepository} className="rounded-lg border border-white/10 bg-zinc-950/70 p-5">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="text-sm font-medium text-zinc-200">
          Owner
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            name="owner"
            placeholder="repository-owner"
            required
          />
        </label>
        <label className="text-sm font-medium text-zinc-200">
          Repository
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            name="repo"
            placeholder="repository-name"
            required
          />
        </label>
        <label className="text-sm font-medium text-zinc-200">
          GitHub repository ID
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            min={1}
            name="github_repository_id"
            placeholder="optional"
            type="number"
          />
        </label>
        <label className="text-sm font-medium text-zinc-200">
          Installation ID
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            min={1}
            name="installation_id"
            placeholder="optional"
            type="number"
          />
        </label>
        <label className="text-sm font-medium text-zinc-200">
          Default branch
          <input
            className="mt-2 h-10 w-full rounded-md border border-white/10 bg-white/[0.03] px-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            name="default_branch"
            placeholder="main"
          />
        </label>
        <label className="flex items-end gap-3 rounded-md border border-white/10 bg-white/[0.03] p-3 text-sm font-medium text-zinc-200">
          <input className="h-4 w-4" name="private" type="checkbox" />
          Private repository
        </label>
      </div>
      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          className="inline-flex h-10 w-fit items-center gap-2 rounded-md bg-zinc-100 px-3 text-sm font-medium text-zinc-950 disabled:cursor-not-allowed disabled:opacity-70"
          disabled={status === "loading"}
          type="submit"
        >
          {status === "loading" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
          {status === "loading" ? "Connecting" : "Connect repository"}
        </button>
        {message ? (
          <p
            className={`flex items-start gap-2 text-sm ${
              status === "success" ? "text-emerald-300" : "text-rose-300"
            }`}
          >
            {status === "success" ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
            )}
            <span>{message}</span>
          </p>
        ) : null}
      </div>
    </form>
  );
}
