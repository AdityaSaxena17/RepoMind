import assert from "node:assert/strict";
import { mock, test } from "node:test";

let repository;
let upsertSnapshot;
let activeJob;
let updates;
let jobs;
let locked;
const prisma = {
  repository: {
    upsert: async () => ({ ...repository, indexedCommitSha: upsertSnapshot }),
    findUnique: async () => {
      assert.equal(locked, true);
      return repository;
    },
    update: async ({ data }) => updates.push(data),
  },
  userRepository: { upsert: async () => ({}) },
  repositoryIndexJob: {
    findFirst: async () => activeJob,
    create: async ({ data }) => {
      jobs.push(data);
      return { id: "job", ...data };
    },
  },
  $queryRaw: async (sql, id) => {
    assert.match(sql.join(""), /FOR UPDATE/);
    assert.equal(id, "repo");
    locked = true;
    return [{ id }];
  },
  $transaction: async (operation) => operation(prisma),
};
mock.module("@clerk/nextjs/server", { namedExports: { auth: {} } });
mock.module("@repo/db", {
  namedExports: {
    prisma,
    Prisma: { TransactionIsolationLevel: { ReadCommitted: "ReadCommitted" } },
    RepositoryAccessStatus: { ACTIVE: "ACTIVE" },
    RepositoryIndexJobStatus: { QUEUED: "QUEUED", RUNNING: "RUNNING" },
    RepositoryIndexJobTrigger: { CONNECT: "CONNECT", RETRY: "RETRY" },
    RepositoryIndexStatus: {
      READY: "READY",
      NOT_INDEXED: "NOT_INDEXED",
      UPDATING: "UPDATING",
      INDEXING: "INDEXING",
    },
    RepositoryVisibility: {
      PUBLIC: "PUBLIC",
      PRIVATE: "PRIVATE",
      INTERNAL: "INTERNAL",
    },
  },
});
const { connectPublicRepositoryForUser } =
  await import("../lib/github-public.ts");

function setup(t, indexedSha) {
  repository = {
    id: "repo",
    indexStatus: indexedSha === null ? "NOT_INDEXED" : "READY",
    indexedCommitSha: indexedSha,
  };
  upsertSnapshot = indexedSha;
  activeJob = null;
  updates = [];
  jobs = [];
  locked = false;
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push(url);
    assert.equal(options.cache, "no-store");
    assert.ok(options.signal instanceof AbortSignal);
    const payload = url.endsWith("/commits/main")
      ? { sha: "a".repeat(40) }
      : {
          id: 1,
          owner: { login: "example" },
          name: "project",
          full_name: "example/project",
          default_branch: "main",
          visibility: "public",
          html_url: "https://github.com/example/project",
        };
    return Response.json(payload);
  });
  return calls;
}

test("matching live and indexed commits creates no indexing job", async (t) => {
  const calls = setup(t, "a".repeat(40));
  const result = await connectPublicRepositoryForUser(
    "user",
    "example",
    "project",
    "RETRY",
  );
  assert.equal(result.job, null);
  assert.equal(result.status, "READY");
  assert.deepEqual(updates, []);
  assert.deepEqual(jobs, []);
  assert.equal(calls.length, 2);
});

test("different or unindexed commits queue exactly one job", async (t) => {
  for (const sha of [null, "b".repeat(40)]) {
    setup(t, sha);
    const result = await connectPublicRepositoryForUser(
      "user",
      "example",
      "project",
      "RETRY",
    );
    assert.equal(result.job.id, "job");
    assert.equal(result.status, sha === null ? "INDEXING" : "UPDATING");
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].sourceCommitSha, sha);
    assert.equal(jobs[0].targetCommitSha, "a".repeat(40));
    assert.equal(updates.length, 1);
  }
});

test("a snapshot promoted during the GitHub check is read after locking", async (t) => {
  setup(t, "a".repeat(40));
  upsertSnapshot = "b".repeat(40);
  const result = await connectPublicRepositoryForUser(
    "user",
    "example",
    "project",
    "RETRY",
  );
  assert.equal(result.job, null);
  assert.deepEqual(jobs, []);
});

test("an existing active job is reused", async (t) => {
  setup(t, "b".repeat(40));
  activeJob = { id: "already-running", status: "RUNNING" };
  const result = await connectPublicRepositoryForUser(
    "user",
    "example",
    "project",
    "RETRY",
  );
  assert.equal(result.job.id, "already-running");
  assert.deepEqual(jobs, []);
});

test("GitHub lookup failures never queue an indexing job", async (t) => {
  setup(t, "b".repeat(40));
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response(null, { status: 503 }),
  );
  await assert.rejects(
    connectPublicRepositoryForUser("user", "example", "project", "RETRY"),
    /503/,
  );
  assert.deepEqual(jobs, []);
  assert.deepEqual(updates, []);
});
