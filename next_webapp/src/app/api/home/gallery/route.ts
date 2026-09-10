import { NextRequest, NextResponse } from "next/server";
import dbConnect from "@/lib/dbConnect";
import GalleryImage from "@/models/mongoose/GalleryImage";

// ── Constants ──────────────────────────────────────────────────────────────
const DEFAULT_PAGE_SIZE = 12;
const MAX_PAGE_SIZE = 100;

function badRequest(message: string) {
  return NextResponse.json({ success: false, message }, { status: 400 });
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}

// ── GET /api/home/gallery ──────────────────────────────────────────────────
// Public endpoint — no authentication required.
// Returns paginated gallery images for the public-facing gallery page.
//
// Query params:
//   page      (number, default 1)
//   pageSize  (number, default 12, max 100)
//   search    (string) — partial, case-insensitive title match
export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const rawPage = url.searchParams.get("page");
    const rawPageSize = url.searchParams.get("pageSize");
    const search = url.searchParams.get("search")?.trim() ?? "";

    if (rawPage !== null && (isNaN(Number(rawPage)) || Number(rawPage) < 1)) {
      return badRequest("page must be a positive integer");
    }
    if (
      rawPageSize !== null &&
      (isNaN(Number(rawPageSize)) || Number(rawPageSize) < 1)
    ) {
      return badRequest("pageSize must be a positive integer");
    }
    if (rawPageSize !== null && Number(rawPageSize) > MAX_PAGE_SIZE) {
      return badRequest(`pageSize cannot exceed ${MAX_PAGE_SIZE}`);
    }

    const page = Math.max(1, parseInt(rawPage ?? "1", 10) || 1);
    const pageSize =
      Math.max(1, parseInt(rawPageSize ?? String(DEFAULT_PAGE_SIZE), 10)) ||
      DEFAULT_PAGE_SIZE;
    const skip = (page - 1) * pageSize;

    await dbConnect();

    const filter: Record<string, unknown> = {};
    if (search) {
      filter.title = { $regex: search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), $options: "i" };
    }

    const [data, total] = await Promise.all([
      GalleryImage.find(filter)
        .select("title img_url created_at")
        .sort({ created_at: -1 })
        .skip(skip)
        .limit(pageSize)
        .lean(),
      GalleryImage.countDocuments(filter),
    ]);

    return NextResponse.json({
      success: true,
      data: data.map(toDTO),
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("[GET /api/home/gallery] unexpected error:", error);
    return serverError("Failed to fetch gallery images");
  }
}
