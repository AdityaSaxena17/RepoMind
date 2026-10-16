"use client";

import { Children, isValidElement, memo, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MermaidDiagram } from "./mermaid-diagram";

/** Render untrusted assistant Markdown; incomplete diagrams remain code while streaming. */
export const MarkdownMessage = memo(function MarkdownMessage({
  content,
  streaming,
}: {
  content: string;
  streaming: boolean;
}) {
  return (
    <div className="markdown-message mt-3 min-w-0 text-sm leading-6 text-zinc-100">
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        disallowedElements={["img"]}
        components={{
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          pre({ children }) {
            const nodes = Children.toArray(children);
            const child = nodes[0];
            if (
              !streaming &&
              nodes.length === 1 &&
              isValidElement<{ className?: string; children?: ReactNode }>(
                child,
              ) &&
              child.props.className
                ?.split(/\s+/)
                .includes("language-mermaid") &&
              typeof child.props.children === "string"
            ) {
              return (
                <MermaidDiagram
                  code={child.props.children.replace(/\n$/, "")}
                />
              );
            }
            return <pre>{children}</pre>;
          },
        }}
      >
        {content}
      </Markdown>
    </div>
  );
});
