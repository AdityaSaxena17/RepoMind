import assert from "node:assert/strict";
import { readFile, writeFile, unlink } from "node:fs/promises";
import { mock, test } from "node:test";
import process from "node:process";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Node's native TypeScript loader excludes JSX; compile the actual component using the project's compiler.
const source = await readFile(
  new URL("../components/markdown-message.tsx", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(
  source.replace('"./mermaid-diagram"', `"./.diagram-${process.pid}.mjs"`),
  {
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext },
  },
).outputText;
const temporary = new URL(`.markdown-${process.pid}.mjs`, import.meta.url);
const diagramTemporary = new URL(
  `.diagram-${process.pid}.mjs`,
  import.meta.url,
);
let MarkdownMessage;
try {
  await writeFile(
    diagramTemporary,
    "export const MermaidDiagram = () => null;\n",
  );
  mock.module(diagramTemporary.href, {
    namedExports: { MermaidDiagram: "mermaid-test-diagram" },
  });
  await writeFile(temporary, compiled);
  ({ MarkdownMessage } = await import(temporary.href));
} finally {
  await unlink(temporary);
  await unlink(diagramTemporary);
}
test("assistant Markdown renders headings, lists, code and GFM tables", () => {
  const html = renderToStaticMarkup(
    createElement(MarkdownMessage, {
      streaming: false,
      content:
        "## Overview\n\n**Strong** and `code`.\n\n- One\n\n| A | B |\n| - | - |\n| 1 | 2 |",
    }),
  );
  for (const tag of ["h2", "strong", "code", "ul", "table"])
    assert.match(html, new RegExp(`<${tag}[ >]`));
});
test("Mermaid blocks render diagrams only after streaming ends", () => {
  const content = "```mermaid\nflowchart TD\n A --> B\n```";
  const pending = renderToStaticMarkup(
    createElement(MarkdownMessage, { streaming: true, content }),
  );
  const complete = renderToStaticMarkup(
    createElement(MarkdownMessage, { streaming: false, content }),
  );
  assert.match(pending, /language-mermaid/);
  assert.doesNotMatch(pending, /mermaid-test-diagram/);
  assert.match(complete, /mermaid-test-diagram/);
});
test("untrusted HTML, remote images and executable links are not rendered", () => {
  const html = renderToStaticMarkup(
    createElement(MarkdownMessage, {
      streaming: false,
      content:
        '<script>alert(1)</script>\n\n<img src="https://example.org/track">\n\n![track](https://example.org/track)\n\n[bad](javascript:alert%281%29)',
    }),
  );
  assert.doesNotMatch(
    html,
    /<script|<img|javascript:|https:\/\/example.org\/track/,
  );
});
