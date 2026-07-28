import { ArrowRight, Bot, Code2, GitBranch, Search } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-svh bg-[#09090b] text-zinc-100">
      <header className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link
          className="flex items-center gap-2 text-sm font-semibold"
          href="/"
        >
          <span className="grid h-8 w-8 place-items-center rounded-md border border-sky-400/30 bg-sky-400/10 text-sky-200">
            <Bot className="h-4 w-4" />
          </span>
          RepoMind
        </Link>
        <nav className="flex items-center gap-3 text-sm">
          <Link href="/sign-in" prefetch={false}>
            Sign in
          </Link>
          <Link
            className="h-9 rounded-md bg-zinc-100 px-4 py-2 font-medium text-zinc-950"
            href="/sign-up"
            prefetch={false}
          >
            Sign up
          </Link>
        </nav>
      </header>
      <section className="mx-auto grid min-h-[calc(100svh-4rem)] max-w-6xl grid-cols-1 gap-10 px-6 py-12 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
        <div>
          <p className="text-sm font-medium text-sky-300">
            AI codebase intelligence
          </p>
          <h1 className="mt-4 max-w-3xl text-5xl font-semibold leading-tight tracking-normal text-zinc-50 md:text-6xl">
            RepoMind
          </h1>
          <p className="mt-5 max-w-xl text-base leading-7 text-zinc-400">
            Ask questions, search intent, inspect GitHub context, and generate
            implementation plans for unfamiliar repositories from one focused
            developer workspace.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              className="inline-flex h-11 items-center gap-2 rounded-md bg-zinc-100 px-5 text-sm font-medium text-zinc-950"
              href="/dashboard"
              prefetch={false}
            >
              Open prototype
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              className="inline-flex h-11 items-center gap-2 rounded-md border border-white/10 px-5 text-sm font-medium text-zinc-200"
              href="/sign-in"
              prefetch={false}
            >
              <GitBranch className="h-4 w-4" />
              Sign in with GitHub
            </Link>
          </div>
        </div>
        <div className="rounded-lg border border-white/10 bg-zinc-950 p-4 shadow-2xl shadow-black/40">
          <div className="flex items-center gap-2 border-b border-white/10 pb-3 text-xs text-zinc-500">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-400" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-300" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
            <span className="ml-2">vercel/commerce</span>
          </div>
          <div className="grid gap-4 pt-4 md:grid-cols-[180px_1fr]">
            <div className="space-y-2 rounded-md border border-white/10 bg-white/[0.03] p-3 text-sm text-zinc-400">
              <p className="flex items-center gap-2 text-zinc-100">
                <Code2 className="h-4 w-4 text-sky-300" />
                Files indexed
              </p>
              <p>1,284 files</p>
              <p>18,492 embeddings</p>
            </div>
            <div className="space-y-3 rounded-md border border-white/10 bg-white/[0.03] p-4">
              <div className="flex items-center gap-2 rounded-md border border-white/10 px-3 py-2 text-sm text-zinc-400">
                <Search className="h-4 w-4" />
                Explain the checkout flow
              </div>
              <p className="text-sm leading-6 text-zinc-300">
                Checkout begins in the cart route, validates line items, then
                delegates payment session creation to the provider adapter.
              </p>
              <div className="flex flex-wrap gap-2 text-xs text-sky-200">
                <span className="rounded-md border border-sky-400/20 bg-sky-400/10 px-2 py-1">
                  app/cart/actions.ts:14-62
                </span>
                <span className="rounded-md border border-sky-400/20 bg-sky-400/10 px-2 py-1">
                  lib/payments/session.ts:8-43
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
