import assert from "node:assert/strict";
import { mock, test } from "node:test";

let found, connections, activeMessages, activeJobs, calls;
const tx = {
  $queryRaw: async (sql, ...ids) => {
    calls.push(["lock", sql.join(""), ids]);
  },
  repository: {
    findFirst: async ({ where }) => {
      assert.deepEqual(where, {
        owner: "owner",
        name: "repo",
        users: { some: { userId: "user" } },
      });
      return found ? { id: "repository" } : null;
    },
    delete: async (args) => calls.push(["repository", args]),
  },
  userRepository: {
    findUnique: async ({ where }) => {
      assert.deepEqual(where, {
        userId_repositoryId: { userId: "user", repositoryId: "repository" },
      });
      return found ? { id: "connection" } : null;
    },
    count: async ({ where }) => {
      assert.deepEqual(where, { repositoryId: "repository" });
      return connections;
    },
    delete: async (args) => calls.push(["connection", args]),
  },
  chatMessage: {
    count: async (args) => {
      calls.push(["active_messages", args]);
      return activeMessages;
    },
  },
  repositoryIndexJob: {
    count: async ({ where }) => {
      assert.deepEqual(where, {
        repositoryId: "repository",
        status: { in: ["QUEUED", "RUNNING"] },
      });
      return activeJobs;
    },
  },
  chatConversation: {
    findFirst: async ({ where }) => {
      assert.deepEqual(where, {
        id: "chat",
        userId: "user",
        repositoryId: "repository",
      });
      return found ? { id: "chat" } : null;
    },
    delete: async (args) => calls.push(["chat", args]),
    deleteMany: async (args) => calls.push(["chats", args]),
  },
};
mock.module("@repo/db", {
  namedExports: {
    prisma: { $transaction: async (operation) => operation(tx) },
  },
});
const { deleteConversationForUser, deleteRepositoryForUser } =
  await import("../lib/deletion.ts");
function setup() {
  found = true;
  connections = 1;
  activeMessages = 0;
  activeJobs = 0;
  calls = [];
}

test("chat deletion scopes both lookup and lock to the caller and repository", async () => {
  setup();
  assert.equal(
    await deleteConversationForUser("user", "repository", "chat"),
    "deleted",
  );
  assert.match(calls[0][1], /"userId"[\s\S]*"repositoryId"[\s\S]*FOR UPDATE/);
  assert.deepEqual(calls[0][2], ["chat", "user", "repository"]);
  assert.deepEqual(calls.at(-1), ["chat", { where: { id: "chat" } }]);
});
test("missing and active chats are never deleted", async () => {
  for (const missing of [true, false]) {
    setup();
    found = !missing;
    activeMessages = 1;
    assert.equal(
      await deleteConversationForUser("user", "repository", "chat"),
      missing ? "not_found" : "busy",
    );
    assert.ok(!calls.some(([name]) => name === "chat"));
  }
});
test("disconnecting a shared repository preserves its index and other accounts' chats", async () => {
  setup();
  connections = 2;
  activeJobs = 1;
  assert.equal(
    await deleteRepositoryForUser("user", "owner", "repo"),
    "deleted",
  );
  assert.deepEqual(
    calls.filter(([name]) =>
      ["chats", "connection", "repository"].includes(name),
    ),
    [
      ["chats", { where: { userId: "user", repositoryId: "repository" } }],
      ["connection", { where: { id: "connection" } }],
    ],
  );
  assert.match(calls[0][1], /FOR UPDATE/);
});
test("the last unused repository connection removes the index under the repository lock", async () => {
  setup();
  assert.equal(
    await deleteRepositoryForUser("user", "owner", "repo"),
    "deleted",
  );
  assert.deepEqual(calls.at(-1), [
    "repository",
    { where: { id: "repository" } },
  ]);
});
test("active jobs or active chats prevent final repository deletion", async () => {
  for (const job of [true, false]) {
    setup();
    activeJobs = job ? 1 : 0;
    activeMessages = job ? 0 : 1;
    assert.equal(
      await deleteRepositoryForUser("user", "owner", "repo"),
      "busy",
    );
    assert.ok(
      !calls.some(([name]) =>
        ["chats", "connection", "repository"].includes(name),
      ),
    );
  }
});
test("an unconnected repository cannot be deleted", async () => {
  setup();
  found = false;
  assert.equal(
    await deleteRepositoryForUser("user", "owner", "repo"),
    "not_found",
  );
  assert.deepEqual(calls, []);
});
test("database failures propagate for controlled route error handling", async (t) => {
  setup();
  t.mock.method(tx.chatConversation, "delete", async () => {
    throw new Error("database unavailable");
  });
  await assert.rejects(
    deleteConversationForUser("user", "repository", "chat"),
    /database unavailable/,
  );
});
