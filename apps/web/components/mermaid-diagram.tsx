"use client";

import Image from "next/image";
import { useEffect, useId, useState } from "react";

let initialized = false;
const MAX_DIAGRAM_LENGTH = 10000;

/** Render a bounded diagram as an SVG image, keeping source visible on failure. */
export function MermaidDiagram({ code }: { code: string }) {
  const id = `mermaid-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const [result, setResult] = useState<{
    code: string;
    image: string | null;
  } | null>(null);
  useEffect(() => {
    if (code.length > MAX_DIAGRAM_LENGTH) return;
    let cancelled = false;
    async function render() {
      try {
        const { default: mermaid } = await import("mermaid");
        if (cancelled) return;
        if (!initialized) {
          mermaid.initialize({
            startOnLoad: false,
            securityLevel: "strict",
            theme: "dark",
            htmlLabels: false,
            flowchart: { htmlLabels: false },
            maxTextSize: MAX_DIAGRAM_LENGTH,
            maxEdges: 100,
            suppressErrorRendering: true,
            secure: [
              "secure",
              "securityLevel",
              "startOnLoad",
              "maxTextSize",
              "maxEdges",
              "suppressErrorRendering",
              "htmlLabels",
              "flowchart",
              "themeCSS",
              "themeVariables",
            ],
          });
          initialized = true;
        }
        const { svg } = await mermaid.render(id, code);
        if (!cancelled)
          setResult({
            code,
            image: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`,
          });
      } catch {
        // Invalid or oversized model output is recoverable; preserve its source.
        if (!cancelled) setResult({ code, image: null });
      }
    }
    void render();
    return () => {
      cancelled = true;
    };
  }, [code, id]);
  const ready = result?.code === code ? result : null;
  const failed = code.length > MAX_DIAGRAM_LENGTH || ready?.image === null;
  return (
    <figure className="my-4 min-w-0 rounded-md border border-white/10 bg-black/20 p-3">
      {ready?.image ? (
        // An image context keeps SVG scripts, links and external resources inert.
        <Image
          src={ready.image}
          alt="Diagram generated from the answer's Mermaid source"
          width={1000}
          height={600}
          unoptimized
          className="h-auto max-w-full"
        />
      ) : (
        <p className="text-xs text-zinc-400" role="status">
          {failed
            ? "Diagram could not be rendered. Its source is available below."
            : "Rendering diagram…"}
        </p>
      )}
      <details className="mt-2 text-xs text-zinc-400" open={failed}>
        <summary className="cursor-pointer">Mermaid source</summary>
        <pre className="mt-2 overflow-x-auto">
          <code>{code}</code>
        </pre>
      </details>
    </figure>
  );
}
