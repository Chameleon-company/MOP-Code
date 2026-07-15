import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import { uploadImageToStorage } from "../library/uploadImageToStorage";

// ── Constants ──────────────────────────────────────────────────────────────
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5 MB
const MAX_TITLE_LENGTH = 200;
const DEFAULT_PAGE_SIZE = 12;
const MAX_PAGE_SIZE = 100;

// ── Auth helpers ───────────────────────────────────────────────────────────
function getUserId(request: NextRequest): number | null {
  const raw = request.headers.get("x-user-id");
  if (!raw) return null;
  const id = Number(raw);
  return Number.isFinite(id) ? id : null;
}

function isAdmin(request: NextRequest): boolean {
  const role = request.headers.get("x-user-role");
  const roleId = request.headers.get("x-user-role-id");
  return role?.toLowerCase() === "admin" || roleId === "1";
}

// ── Response helpers ───────────────────────────────────────────────────────
function unauthorized() {
  return NextResponse.json({ success: false, message: "Unauthorised" }, { status: 401 });
}

function forbidden() {
  return NextResponse.json(
    { success: false, message: "Forbidden - Admin only" },
    { status: 403 }
  );
}

function badRequest(message: string, errors?: Record<string, string>) {
  return NextResponse.json(
    { success: false, message, ...(errors && { errors }) },
    { status: 400 }
  );
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// ── Validation ─────────────────────────────────────────────────────────────
function validateTitle(title: string | undefined): string | null {
  if (!title || title.trim().length === 0) return "Title is required";
  if (title.length > MAX_TITLE_LENGTH)
    return `Title must be ${MAX_TITLE_LENGTH} characters or fewer`;
  return null;
}

function validateImage(image: File | null): string | null {
  if (!image || image.size === 0) return "Image file is required";
  if (!ALLOWED_TYPES.includes(image.type))
    return "Only JPEG, PNG, or WebP images are allowed";
  if (image.size > MAX_IMAGE_SIZE) return "Image must be under 5 MB";
  return null;
}

// ── GET /api/gallery ───────────────────────────────────────────────────────
// Admin listing with pagination and optional title search.
// Query params: page, pageSize, search
export async function GET(request: NextRequest) {
  if (!getUserId(request)) return unauthorized();

  try {
    const url = new URL(request.url);
    const rawPage = url.searchParams.get("page");
    const rawPageSize = url.searchParams.get("pageSize");
    const search = url.searchParams.get("search")?.trim() ?? "";

    // Validate pagination params
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
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;

    let query = supabase
      .from("gallery_images")
      .select("id, title, img_url, created_at, created_by", { count: "exact" })
      .order("created_at", { ascending: false });

    if (search) {
      query = query.ilike("title", `%${search}%`);
    }

    const { data, error, count } = await query.range(from, to);

    if (error) {
      console.error("[GET /api/gallery] error:", error);
      return serverError("Failed to fetch gallery images");
    }

    const total = count ?? 0;

    return NextResponse.json({
      success: true,
      data: data ?? [],
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("[GET /api/gallery] unexpected error:", error);
    return serverError("Failed to fetch gallery images");
  }
}

// ── POST /api/gallery ──────────────────────────────────────────────────────
// Admin only. Accepts multipart/form-data: title (string), image (File).
export async function POST(request: NextRequest) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  try {
    const formData = await request.formData();
    const title = formData.get("title")?.toString().trim();
    const image = formData.get("image") as File | null;

    const errors: Record<string, string> = {};
    const titleErr = validateTitle(title);
    const imageErr = validateImage(image);
    if (titleErr) errors.title = titleErr;
    if (imageErr) errors.image = imageErr;

    if (Object.keys(errors).length > 0) {
      return badRequest("Validation failed", errors);
    }

    const uploaded = await uploadImageToStorage({
      file: image as File,
      bucket: "gallery-images",
      folder: "gallery",
      prefix: "gallery",
      userId,
    });

    const { data: galleryImage, error } = await supabase
      .from("gallery_images")
      .insert({
        title,
        img_url: uploaded.publicUrl,
        created_by: userId,
      })
      .select("id, title, img_url, created_at, created_by")
      .single();

    if (error) {
      console.error("[POST /api/gallery] insert error:", error);
      return serverError("Failed to create gallery image");
    }

    return NextResponse.json(
      {
        success: true,
        message: "Gallery image added successfully",
        data: galleryImage,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("[POST /api/gallery] unexpected error:", error);
    const message =
      error instanceof Error ? error.message : "Failed to add gallery image";
    return serverError(message);
  }
}
