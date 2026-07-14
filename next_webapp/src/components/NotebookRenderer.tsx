"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { useState } from "react";

type Source = string | string[];

interface NotebookOutput {
  output_type: "stream" | "display_data" | "execute_result" | "error";
  text?: Source;
  data?: Record<string, Source>;
  ename?: string;
  evalue?: string;
}

interface NotebookCell {
  cell_type: "markdown" | "code" | "raw";
  source: Source;
  outputs?: NotebookOutput[];
}

interface Notebook {
  cells: NotebookCell[];
}

function join(src: Source): string {
  return Array.isArray(src) ? src.join("") : (src ?? "");
}

function CellOutput({ output }: { output: NotebookOutput }) {
  if (output.output_type === "stream") {
    const text = join(output.text ?? "");
    if (!text) return null;
    return (
      <pre className="overflow-x-auto rounded-lg bg-gray-50 p-4 text-sm leading-relaxed text-gray-800 dark:bg-gray-900 dark:text-gray-200">
        {text}
      </pre>
    );
  }

  if (
    output.output_type === "display_data" ||
    output.output_type === "execute_result"
  ) {
    const data = output.data ?? {};

    if (data["image/png"]) {
      return (
        <img
          src={`data:image/png;base64,${join(data["image/png"]).trim()}`}
          alt="notebook output"
          className="max-w-full rounded-lg"
        />
      );
    }
    if (data["image/jpeg"]) {
      return (
        <img
          src={`data:image/jpeg;base64,${join(data["image/jpeg"]).trim()}`}
          alt="notebook output"
          className="max-w-full rounded-lg"
        />
      );
    }
    if (data["image/svg+xml"]) {
      return (
        <div
          className="overflow-x-auto"
          dangerouslySetInnerHTML={{ __html: join(data["image/svg+xml"]) }}
        />
      );
    }
    if (data["text/html"]) {
      // Strip URL-based src from iframes (e.g. IPython.display.IFrame embeds)
      // to prevent page-in-page renders like "Use case not found".
      // data: URIs are preserved so Folium/Plotly maps still render.
      const rawHtml = join(data["text/html"]);
      const safeHtml = rawHtml.replace(
        /(<iframe\b[^>]*?)\s+src="(?!data:)[^"]*"/gi,
        "$1"
      );
      return (
        <div
          className="overflow-x-auto"
          dangerouslySetInnerHTML={{ __html: safeHtml }}
        />
      );
    }
    if (data["text/plain"]) {
      return (
        <pre className="overflow-x-auto rounded-lg bg-gray-50 p-4 text-sm text-gray-800 dark:bg-gray-900 dark:text-gray-200">
          {join(data["text/plain"])}
        </pre>
      );
    }
  }

  if (output.output_type === "error") {
    return (
      <pre className="overflow-x-auto rounded-lg bg-red-50 p-4 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
        {output.ename}: {output.evalue}
      </pre>
    );
  }

  return null;
}

function CodeCell({ cell, showCode }: { cell: NotebookCell; showCode: boolean }) {
  const outputs = cell.outputs ?? [];
  const source = join(cell.source);
  const hasOutputs = outputs.length > 0;
  const hasSource = source.trim().length > 0;

  if (!hasOutputs && !hasSource) return null;

  return (
    <div className="space-y-2">
      {showCode && hasSource && (
        <pre className="overflow-x-auto rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-800 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200">
          <code>{source}</code>
        </pre>
      )}
      {hasOutputs && (
        <div className="space-y-2">
          {outputs.map((output, i) => (
            <CellOutput key={i} output={output} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function NotebookRenderer({ content }: { content: string }) {
  const [showCode, setShowCode] = useState(false);

  let notebook: Notebook;
  try {
    notebook = JSON.parse(content);
  } catch {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        Invalid notebook format.
      </p>
    );
  }

  if (!Array.isArray(notebook?.cells)) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        No content found in notebook.
      </p>
    );
  }

  const hasCode = notebook.cells.some(
    (c) => c.cell_type === "code" && join(c.source).trim().length > 0
  );

  return (
    <div>
      {hasCode && (
        <div className="mb-4 flex justify-end">
          <button
            onClick={() => setShowCode((v) => !v)}
            className="rounded-full border border-gray-200 px-4 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-50 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-800"
          >
            {showCode ? "Hide code" : "Show code"}
          </button>
        </div>
      )}

      <div className="space-y-6">
        {notebook.cells.map((cell, i) => {
          if (cell.cell_type === "markdown") {
            return (
              <div
                key={i}
                className="prose prose-gray max-w-none dark:prose-invert"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                  {join(cell.source)}
                </ReactMarkdown>
              </div>
            );
          }

          if (cell.cell_type === "code") {
            return <CodeCell key={i} cell={cell} showCode={showCode} />;
          }

          return null;
        })}
      </div>
    </div>
  );
}
