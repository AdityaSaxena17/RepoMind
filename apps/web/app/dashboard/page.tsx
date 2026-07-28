import { auth } from "@clerk/nextjs/server";
import { prisma } from "@repo/db";

export default async function Dashboard() {
  const { userId } = await auth.protect();
  const user = await prisma.user.findUnique({
    where: {
      clerkId: userId,
    },
  });

  return (
    <main className="mx-auto flex min-h-[calc(100svh-73px)] max-w-3xl flex-col justify-center px-6 py-12">
      <p className="text-sm font-medium text-repo-blue">Signed in</p>
      <h1 className="mt-3 text-4xl font-semibold tracking-normal">
        Welcome to RepoMind.
      </h1>
      <p className="mt-4 max-w-xl text-base leading-7 text-black/70 dark:text-white/70">
        This page is protected by Clerk. Your active Clerk user ID is{" "}
        <code className="rounded bg-black/5 px-1.5 py-1 text-sm font-semibold dark:bg-white/10">
          {userId}
        </code>
        .
      </p>
      <div className="mt-8 rounded-lg border border-black/10 p-5 dark:border-white/15">
        <p className="text-sm font-semibold">
          Database sync: {user ? "synced" : "waiting for webhook"}
        </p>
        <p className="mt-2 text-sm leading-6 text-black/70 dark:text-white/70">
          {user
            ? `Stored as ${user.email ?? user.clerkId} with ${user.status.toLowerCase()} status.`
            : "Create or replay a Clerk user webhook to write this account into Postgres."}
        </p>
      </div>
    </main>
  );
}
