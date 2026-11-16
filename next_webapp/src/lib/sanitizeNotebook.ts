
import DOMPurify from "isomorphic-dompurify";

export function sanitizeNotebookSvg(input: string): string {
  return DOMPurify.sanitize(input, {
    USE_PROFILES: {
      svg: true,
      svgFilters: true,
    },
    FORBID_TAGS: ["foreignObject"],
  });
}

export function sanitizeNotebookHtml(input: string): string {
  return DOMPurify.sanitize(input, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["iframe", "object", "embed"],
    FORBID_ATTR: ["srcdoc"],
  });
}
