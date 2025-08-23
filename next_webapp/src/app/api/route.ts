import { promises as fs } from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";          // fs requires Node runtime
export const dynamic = "force-dynamic";   // always run on server

// ----- Types -----
type Item = {
  id: string;
  name: string;           // contributor name OR project name
  email?: string;         // contributor email (for contributions)
  title: string;          // dataset title
  description: string;
  tags: string[];
  filename: string;
  uploadedAt?: string;
  source: "published" | "contribution";
};

// ----- Paths -----
const ROOT = process.cwd();
const PUBLISHED_BASE = path.join(ROOT, "datascience", "usecases", "READY TO PUBLISH");
// New inbox for uploads:
const INBOX_BASE     = path.join(ROOT, "datascience", "contributions", "INBOX");

// ----- Upload policy -----
const ALLOWED_MIME = new Set([
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/json",
]);
const MAX_MB = 20;

// ----- Helpers -----
const sanitize = (s: string) => s.replace(/[^\w\-]+/g, "_").slice(0, 80);

async function listItems(base: string, source: Item["source"]): Promise<Item[]> {
  const out: Item[] = [];
  let entries: any[] = [];
  try {
    entries = await fs.readdir(base, { withFileTypes: true });
  } catch {
    return out; // folder may not exist yet
  }

  for (const dirent of entries) {
    if (!dirent.isDirectory()) continue;
    const folder = dirent.name;
    const metaFile = path.join(base, folder, "metadata.json");
    try {
      const raw = await fs.readFile(metaFile, "utf-8");
      const m = JSON.parse(raw);

      out.push({
        id: folder,
        name: m.name ?? "",                    // for contributions this is contributor name
        email: m.email ?? undefined,
        title: m.title ?? m.name ?? "",
        description: m.description ?? "",
        tags: m.tags ?? [],
        filename: m.filename ?? "",
        uploadedAt: m.uploadedAt ?? undefined,
        source,
      });
    } catch {
      // skip invalid folders
    }
  }
  return out;
}

// ===== GET: list published + contributions (with ?q= filter) =====
export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const q = (url.searchParams.get("q") || "").trim().toLowerCase();

  const [published, contributions] = await Promise.all([
    listItems(PUBLISHED_BASE, "published"),
    listItems(INBOX_BASE, "contribution"),
  ]);

  let results = [...published, ...contributions];

  if (q) {
    results = results.filter((u) =>
      [u.title, u.description, ...(u.tags || [])]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }

  return NextResponse.json({ results });
}

// ===== POST: accept upload from Contribute form =====
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();

    const name = String(form.get("name") || "");
    const email = String(form.get("email") || "");
    const title = String(form.get("title") || "");
    const description = String(form.get("description") || "");
    const tags = String(form.get("tags") || "")
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const file = form.get("file") as File | null;

    if (!name || !email || !title || !description || !file) {
      return NextResponse.json({ ok: false, error: "Missing fields" }, { status: 400 });
    }

    if (!ALLOWED_MIME.has(file.type)) {
      return NextResponse.json({ ok: false, error: "Unsupported file type" }, { status: 400 });
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return NextResponse.json({ ok: false, error: `File too large (> ${MAX_MB} MB)` }, { status: 400 });
    }

    // Ensure base exists
    await fs.mkdir(INBOX_BASE, { recursive: true });

    // Unique folder for this submission
    const id = `${Date.now()}_${sanitize(title)}`;
    const targetDir = path.join(INBOX_BASE, id);
    await fs.mkdir(targetDir, { recursive: true });

    // Derive filename + write file
    const originalName = sanitize((file as any).name || "dataset");
    const extFromName = path.extname(originalName);
    const ext =
      extFromName ||
      (file.type === "application/json"
        ? ".json"
        : file.type.includes("csv")
        ? ".csv"
        : file.type.includes("sheet")
        ? ".xlsx"
        : "");
    const filename = extFromName ? originalName : originalName + ext;

    const buf = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(path.join(targetDir, filename), buf);

    // Write metadata.json
    const metadata = {
      name,
      email,
      title,
      description,
      tags,
      filename,
      uploadedAt: new Date().toISOString(),
    };
    await fs.writeFile(
      path.join(targetDir, "metadata.json"),
      JSON.stringify(metadata, null, 2),
      "utf-8"
    );

    return NextResponse.json({ ok: true, id }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json(
      { ok: false, error: err?.message || "Server error" },
      { status: 500 }
    );
  }
}
