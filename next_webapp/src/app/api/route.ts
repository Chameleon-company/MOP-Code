import { promises as fs } from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

interface UseCase {
  id: string;
  name: string;
  description: string;
  tags: string[];
  filename: string;
}

const BASE = path.join(
  process.cwd(),
  "datascience",
  "usecases",
  "READY TO PUBLISH"
);

export async function GET(req: NextRequest) {
  // read all folder names under READY TO PUBLISH
  const entries = await fs.readdir(BASE, { withFileTypes: true });
  const all: UseCase[] = [];

  for (const dirent of entries) {
    if (!dirent.isDirectory()) continue;
    const folder = dirent.name;
    const metaFile = path.join(BASE, folder, "metadata.json");

    try {
      const raw = await fs.readFile(metaFile, "utf-8");
      const { name, description, tags, filename } = JSON.parse(raw);

      all.push({
        id: folder,
        name,
        description,
        tags,
        filename,
      });
    } catch (err) {
      // skip folders without valid metadata.json
      console.warn(`Skipping ${folder}: no valid metadata.json`);
    }
  }

  // check for an incoming ?q=keyword filter
  const url = new URL(req.url);
  const q = (url.searchParams.get("q") || "").trim().toLowerCase();

  const results = q
    ? all.filter((u) =>
        [u.name, u.description, ...u.tags]
          .join(" ")
          .toLowerCase()
          .includes(q)
      )
    : all;

  return NextResponse.json({ results });
}


