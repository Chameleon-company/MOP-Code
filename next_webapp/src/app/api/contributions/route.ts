import { promises as fs } from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";        // FS needs Node runtime (not Edge)
export const dynamic = "force-dynamic"; // don't cache

// Where uploads are stored
const INBOX_BASE = path.join(
  process.cwd(),
  "datascience",
  "contributions",
  "INBOX"
);

// Allowed file types / size
const ALLOWED_MIME = new Set([
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/json",
]);
const MAX_MB = 20;

const sanitize = (s: string) => s.replace(/[^\w\-]+/g, "_").slice(0, 80);

// ---------- GET: list contributions (supports ?q=search) ----------
export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const q = (url.searchParams.get("q") || "").trim().toLowerCase();

  const items: any[] = [];
  let entries: any[] = [];
  try {
    entries = await fs.readdir(INBOX_BASE, { withFileTypes: true });
  } catch {
    // folder may not exist yet
  }

  for (const dirent of entries) {
    if (!dirent.isDirectory()) continue;
    const folder = dirent.name;
    const metaFile = path.join(INBOX_BASE, folder, "metadata.json");
    try {
      const raw = await fs.readFile(metaFile, "utf-8");
      const m = JSON.parse(raw);
      items.push({
        id: folder,
        name: m.name ?? "",
        email: m.email ?? "",
        title: m.title ?? "",
        description: m.description ?? "",
        tags: m.tags ?? [],
        filename: m.filename ?? "",
        uploadedAt: m.uploadedAt ?? "",
        source: "contribution",
      });
    } catch {
      // skip invalid
    }
  }

  const results = q
    ? items.filter((i) =>
        [i.title, i.description, ...(i.tags || [])]
          .join(" ")
          .toLowerCase()
          .includes(q)
      )
    : items;

  return NextResponse.json({ results });
}

// ---------- POST: accept uploads from /[locale]/upload ----------
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

    // Ensure base folder
    await fs.mkdir(INBOX_BASE, { recursive: true });

    // Unique folder per submission
    const id = `${Date.now()}_${sanitize(title)}`;
    const targetDir = path.join(INBOX_BASE, id);
    await fs.mkdir(targetDir, { recursive: true });

    // Save file
    const originalName = sanitize((file as any).name || "dataset");
    const ab = await file.arrayBuffer();               
    const bytes = new Uint8Array(ab);                    
    await fs.writeFile(path.join(targetDir, originalName), bytes);

    // Save metadata
    const metadata = {
      name,
      email,
      title,
      description,
      tags,
      filename: originalName,
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
