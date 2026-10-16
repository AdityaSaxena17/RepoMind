import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));
const browser = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/chromium",
  "/usr/bin/google-chrome",
].find(existsSync);

test(
  "chat retains streamed messages across history and completion",
  { skip: !browser, timeout: 90000 },
  async (t) => {
    // Exercise the actual component in a real browser, using the installed Next compiler.
    const require = createRequire(new URL("../package.json", import.meta.url));
    const { webpack } = require("next/dist/compiled/webpack/webpack");
    const temporary = await mkdtemp(path.join(tmpdir(), "repomind-chat-ui-"));
    const output = path.join(temporary, "output");
    const loader = path.join(temporary, "typescript-loader.cjs");
    const entry = path.join(temporary, "entry.tsx");
    await writeFile(
      loader,
      `const ts = require(${JSON.stringify(require.resolve("typescript"))});
    module.exports = function(source) { return ts.transpileModule(source, {
      compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 }
    }).outputText; };`,
    );
    // The test uses image-context SVGs too, without depending on a Next server.
    const image = path.join(temporary, "image.tsx");
    await writeFile(
      image,
      `import { createElement } from "react";
    export default function Image({unoptimized, ...props}) { return createElement("img", props); }`,
    );
    await writeFile(
      entry,
      `import { createElement } from "react";
    import { createRoot } from "react-dom/client";
    import { RepositoryChat } from ${JSON.stringify(path.join(webRoot, "components/repository-chat.tsx"))};
    const scenario = new URLSearchParams(location.search).get("scenario");
    const conversation = {id:"conversation",title:"Existing chat",created_at:"2026-01-01",updated_at:"2026-01-01",messages:[]};
    window.gets = 0;
    window.posts = [];
    window.fetch = async (_url, options = {}) => {
      if (options.method === "POST") {
        window.posts.push(JSON.parse(options.body));
        return new Response(new ReadableStream({start(controller) {window.answerStream = controller;}}), {
          headers: {"X-RepoMind-Conversation-Id":JSON.parse(options.body).conversation_id ?? "new-conversation", "X-RepoMind-User-Message-Id":"saved-user"}
        });
      }
      window.gets++;
      if (scenario === "late-history") await new Promise(resolve => {window.releaseHistory = resolve;});
      return Response.json({conversations:[conversation]});
    };
    window.emit = (event, data) => window.answerStream.enqueue(new TextEncoder().encode("event: " + event + "\\ndata: " + JSON.stringify(data) + "\\n\\n"));
    createRoot(document.getElementById("root")).render(createElement(RepositoryChat, {
      repository:{id:"repository",owner:"owner",name:"repo",status:"READY",embeddings:1,indexedFiles:1,branch:"main",htmlUrl:null}
    }));`,
    );
    const compiler = webpack({
      mode: "development",
      target: "web",
      entry,
      output: { path: output, filename: "chat.js", publicPath: "/" },
      resolve: {
        extensions: [".tsx", ".ts", ".js"],
        modules: [path.join(webRoot, "node_modules"), "node_modules"],
        alias: { "next/image": image },
      },
      module: {
        rules: [{ test: /\.tsx?$/, exclude: /node_modules/, use: loader }],
      },
      devtool: false,
    });
    try {
      await new Promise((resolve, reject) =>
        compiler.run((error, stats) => {
          if (error || stats.hasErrors())
            reject(
              error ?? new Error(stats.toString({ all: false, errors: true })),
            );
          else resolve();
        }),
      );
    } finally {
      await new Promise((resolve, reject) =>
        compiler.close((error) => (error ? reject(error) : resolve())),
      );
    }
    const cssDirectory = path.join(webRoot, ".next/static/chunks");
    const styles = existsSync(cssDirectory)
      ? (
          await Promise.all(
            (await readdir(cssDirectory))
              .filter((name) => name.endsWith(".css"))
              .map((name) => readFile(path.join(cssDirectory, name), "utf8")),
          )
        ).join("\n")
      : "";
    const server = createServer(async (request, response) => {
      try {
        const pathname = new URL(request.url, "http://localhost").pathname;
        if (pathname === "/") {
          response.setHeader("Content-Type", "text/html");
          response.end(
            `<html><head><style>${styles}</style></head><body style="margin:0"><div id="root"></div><script src="/chat.js"></script></body></html>`,
          );
        } else {
          const file = path.resolve(output, `.${pathname}`);
          if (!file.startsWith(output + path.sep)) {
            response.writeHead(403).end();
            return;
          }
          response.setHeader("Content-Type", "text/javascript");
          response.end(await readFile(file));
        }
      } catch {
        response.writeHead(404).end();
      }
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const child = spawn(
      browser,
      [
        "--headless",
        "--remote-debugging-port=0",
        `--user-data-dir=${temporary}/profile`,
        "--disable-background-networking",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
      ],
      { windowsHide: true, stdio: "ignore" },
    );
    let socket;
    let command = 0;
    const pending = new Map();
    async function poll(check) {
      const deadline = Date.now() + 10000;
      while (Date.now() < deadline) {
        if (await check()) return;
        await delay(25);
      }
      throw new Error("Timed out waiting for browser state.");
    }
    async function send(method, params = {}) {
      const id = ++command;
      const result = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`Browser command timed out: ${method}`));
        }, 5000);
        pending.set(id, {
          resolve: (value) => {
            clearTimeout(timeout);
            resolve(value);
          },
          reject: (error) => {
            clearTimeout(timeout);
            reject(error);
          },
        });
      });
      socket.send(JSON.stringify({ id, method, params }));
      return result;
    }
    async function evaluate(expression) {
      const result = await send("Runtime.evaluate", {
        expression,
        returnByValue: true,
        awaitPromise: true,
      });
      if (result.exceptionDetails)
        throw new Error(result.exceptionDetails.text);
      return result.result.value;
    }
    async function open(scenario) {
      await send("Page.navigate", {
        url: `http://127.0.0.1:${server.address().port}/?scenario=${scenario}`,
      });
      await poll(() =>
        evaluate("Boolean(document.querySelector('textarea') && window.gets)"),
      );
      if (scenario !== "late-history")
        await poll(() =>
          evaluate(
            "!document.body.textContent.includes('Loading chat history')",
          ),
        );
    }
    async function startAnswer() {
      await evaluate(`(() => {const input=document.querySelector("textarea");
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,"value").set.call(input,"Explain the implementation");
      input.dispatchEvent(new Event("input",{bubbles:true}));})()`);
      await poll(() =>
        evaluate(
          "!document.querySelector('textarea').nextElementSibling.disabled",
        ),
      );
      await evaluate(
        "document.querySelector('textarea').nextElementSibling.click()",
      );
      await poll(() => evaluate("Boolean(window.answerStream)"));
      await evaluate(
        `window.emit("assistant_message",{message_id:"answer"}); window.emit("delta",{text:"Observed answer [1].",native:true})`,
      );
      await poll(() =>
        evaluate(
          "document.querySelectorAll('article').length === 2 && document.body.textContent.includes('Observed answer')",
        ),
      );
    }
    try {
      child.on("error", (error) => {
        for (const task of pending.values()) task.reject(error);
      });
      const portFile = path.join(temporary, "profile/DevToolsActivePort");
      await poll(() => existsSync(portFile));
      const [port] = (await readFile(portFile, "utf8")).split("\n");
      const targets = await (
        await fetch(`http://127.0.0.1:${port}/json/list`)
      ).json();
      socket = new WebSocket(
        targets.find((target) => target.type === "page").webSocketDebuggerUrl,
      );
      await once(socket, "open");
      socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        const task = pending.get(message.id);
        if (task) {
          pending.delete(message.id);
          if (message.error) task.reject(new Error(message.error.message));
          else task.resolve(message.result);
        }
      });
      await t.test(
        "completion keeps streamed content and citations without replacing history",
        async () => {
          await open("complete");
          await startAnswer();
          await evaluate(
            `window.emit("citations",{citations:[{id:"citation",source_index:1,path:"app.py",start_line:1,end_line:2,url:"https://github.com/example/project/blob/commit/app.py#L1-L2"}]});window.emit("done",{message_id:"answer"})`,
          );
          await poll(() =>
            evaluate(
              "document.body.textContent.includes('Response validated and saved')",
            ),
          );
          await delay(100);
          assert.equal(
            await evaluate("document.querySelectorAll('article').length"),
            2,
          );
          assert.equal(await evaluate("window.gets"), 1);
          assert.equal(
            await evaluate("document.body.textContent.includes('app.py:1-2')"),
            true,
          );
        },
      );
      await t.test(
        "late history cannot erase a live partial answer",
        async () => {
          await open("late-history");
          await startAnswer();
          await evaluate("window.releaseHistory()");
          await delay(100);
          assert.equal(
            await evaluate("document.querySelectorAll('article').length"),
            2,
          );
          assert.equal(
            await evaluate(
              "document.body.textContent.includes('Observed answer')",
            ),
            true,
          );
        },
      );
      await t.test(
        "stream errors retain partial text with a visible failure",
        async () => {
          await open("error");
          await startAnswer();
          await evaluate(
            `window.emit("error",{message:"Controlled generation failure"})`,
          );
          await poll(() =>
            evaluate(
              "document.body.textContent.includes('Partial text is unverified')",
            ),
          );
          assert.equal(
            await evaluate("document.querySelectorAll('article').length"),
            2,
          );
          assert.equal(
            await evaluate(
              "document.body.textContent.includes('Observed answer')",
            ),
            true,
          );
        },
      );
      await t.test(
        "starting a new chat preserves other conversations",
        async () => {
          await open("new-chat");
          await evaluate(
            "Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'New chat').click()",
          );
          await startAnswer();
          assert.equal(
            await evaluate(
              "Array.from(document.querySelectorAll('button')).some(button => button.textContent === 'Existing chat')",
            ),
            true,
          );
        },
      );
    } finally {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ id: ++command, method: "Browser.close" }));
        await Promise.race([once(socket, "close"), delay(5000)]);
        socket.close();
      }
      if (child.exitCode === null) child.kill();
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
      // Cleanup is limited to the unique test directory created above.
      assert.equal(
        path.dirname(path.resolve(temporary)),
        path.resolve(tmpdir()),
      );
      await rm(temporary, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 100,
      });
    }
  },
);
