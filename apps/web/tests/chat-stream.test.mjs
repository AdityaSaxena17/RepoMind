import assert from "node:assert/strict";
import { test } from "node:test";
import { readSseBlocks } from "../lib/chat-stream.ts";

test(
  "the client can finish on SSE done while the connection stays open",
  { timeout: 1000 },
  async () => {
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            'event: delta\ndata: {"text":"Hello","native":true}\n\nevent: done\ndata: {"message_id":"answer"}\n\n',
          ),
        );
        // A proxy is allowed to leave the HTTP connection open after SSE done.
      },
    });
    const events = [];
    for await (const block of readSseBlocks(body)) {
      events.push(block);
      if (block.startsWith("event: done")) break;
    }
    assert.equal(events.length, 2);
    assert.equal(body.locked, false);
    await body.cancel();
  },
);

test("SSE and UTF-8 tokens survive arbitrary network chunk boundaries", async () => {
  const expected = 'event: delta\ndata: {"text":"café","native":true}';
  const bytes = new TextEncoder().encode(
    expected.replaceAll("\n", "\r\n") + "\r\n\r\n",
  );
  const body = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  });
  const blocks = [];
  for await (const block of readSseBlocks(body)) blocks.push(block);
  assert.deepEqual(blocks, [expected]);
  assert.equal(body.locked, false);
});

test("stream failures surface and release the reader", async () => {
  const body = new ReadableStream({
    start(controller) {
      controller.error(new Error("Disconnected"));
    },
  });
  await assert.rejects(async () => {
    for await (const block of readSseBlocks(body)) assert.fail(block);
  }, /Disconnected/);
  assert.equal(body.locked, false);
});
