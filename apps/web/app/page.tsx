import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-[calc(100svh-73px)] max-w-3xl flex-col justify-center px-6 py-12">
      <p className="text-sm font-medium text-repo-blue">RepoMind</p>
      <h1 className="mt-3 text-4xl font-semibold tracking-normal">
        GitHub-powered auth is ready to wire up.
      </h1>
      <p className="mt-4 max-w-xl text-base leading-7 text-black/70 dark:text-white/70">
        Sign in with Clerk, then head to the protected dashboard to verify the
        active session.
      </p>
      <Link
        className="mt-8 flex h-11 w-fit items-center rounded-full bg-[var(--foreground)] px-5 text-sm font-medium text-[var(--background)]"
        href="/dashboard"
        prefetch={false}
      >
        Open dashboard
      </Link>
    </main>
  );
}
