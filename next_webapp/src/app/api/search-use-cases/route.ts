export const runtime = 'nodejs';

import { CaseStudy, SEARCH_MODE, SearchParams } from "@/app/types";
import fs from "fs";
import path from "path";

export async function POST(request: Request) {
  // Path to datascience/usecases/READY TO PUBLISH
  const baseDir = path.join(
    process.cwd(),
    "..",
    "datascience",
    "usecases",
    "READY TO PUBLISH"
  );

  const { category, searchMode, searchTerm } = (await request.json()) as SearchParams;
  console.log("~ POST ~ searchParams:", { category, searchMode, searchTerm });

  // Normalize & strip “use case…” boilerplate
  const raw = searchTerm.trim().toLowerCase();
  const term = raw.replace(
    /^(?:show\s+)?use\s+cases?(?:\s+(?:about|on|for))?\s*/i,
    ""
  );

  // Load all case-study folders
  const caseStudies: CaseStudy[] = [];
  let id = 1;

  if (fs.existsSync(baseDir)) {
    for (const termFolder of fs
      .readdirSync(baseDir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)) 
    {
      const termPath = path.join(baseDir, termFolder);

      for (const subdir of fs
        .readdirSync(termPath, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)) 
      {
        const subdirPath = path.join(termPath, subdir);
        const files = fs.readdirSync(subdirPath);
        const jsonFile = files.find((f) => f.endsWith(".json"));
        const htmlFile = files.find((f) => f.endsWith(".html"));
        if (!jsonFile || !htmlFile) continue;

        try {
          const rawJson = fs.readFileSync(
            path.join(subdirPath, jsonFile),
            "utf-8"
          );
          const { name, description, tags } = JSON.parse(rawJson);
          caseStudies.push({
            id: id++,
            name,
            description,
            tags,
          });
        } catch (e) {
          console.warn("Skipping malformed JSON in", subdirPath, e);
        }
      }
    }
  }

  // Filter results
  const genericQueries = [
    "use cases",
    "what are use cases",
    "show me use cases",
    "list use cases",
  ];

  let filteredStudies: CaseStudy[] = [];

  if (genericQueries.includes(raw) || term === "") {
    // Generic catch-all → return first five
    filteredStudies = caseStudies.slice(0, 5);
  } else if (searchMode === SEARCH_MODE.TITLE) {
    filteredStudies = caseStudies.filter((cs) =>
      cs.name.toLowerCase().includes(term)
    );
  } else {
    // CONTENT: search name, description, AND tags
    filteredStudies = caseStudies.filter((cs) =>
      cs.name.toLowerCase().includes(term) ||
      cs.description.toLowerCase().includes(term) ||
      cs.tags.some((t) => t.toLowerCase().includes(term))
    );
  }

  console.log(" ~ POST ~ filteredStudies:", filteredStudies);
  return Response.json({ filteredStudies });
}
