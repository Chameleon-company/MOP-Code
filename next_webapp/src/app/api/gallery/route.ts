import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import GalleryImage from "@/models/mongoose/GalleryImage";
import { uploadImageToGCS } from "../library/uploadImageToGCS";

const GCS_IMAGES_BUCKET = process.env.GCS_IMAGES_BUCKET ?? "mop-images";

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

// created_by is a Mongo ObjectId ref — only usable once the header carries a
// real Mongo User _id (post Auth-phase migration). Until then, fall back to
// null rather than let Mongoose throw a CastError on a legacy numeric id.
function getCreatedBy(request: NextRequest): string | null {
  const raw = request.headers.get("x-user-id");
  return raw && mongoose.Types.ObjectId.isValid(raw) ? raw : null;
}

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
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
  const { userId, isAuthenticated } = getAuthUser(request);

  if (!isAuthenticated || !userId) return unauthorized();

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
    const skip = (page - 1) * pageSize;

    await dbConnect();

    const filter: Record<string, unknown> = {};
    if (search) {
      filter.title = { $regex: search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), $options: "i" };
    }

    const [data, total] = await Promise.all([
      GalleryImage.find(filter)
        .select("title img_url created_at created_by")
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
    console.error("[GET /api/gallery] unexpected error:", error);
    return serverError("Failed to fetch gallery images");
  }
}

// ── POST /api/gallery ──────────────────────────────────────────────────────
// Admin only. Accepts multipart/form-data: title (string), image (File).
export async function POST(request: NextRequest) {
  const { userId, isAuthenticated, isAdmin } = getAuthUser(request);

  if (!isAuthenticated || !userId) {
    return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request, userId);
  }

  if (!isAdmin) {
    return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request, userId);
  }

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

    const buffer = Buffer.from(await (image as File).arrayBuffer());
    const filename = `gallery/gallery-${userId}-${Date.now()}.webp`;

    let imgUrl: string;
    try {
      imgUrl = await uploadImageToGCS(buffer, filename, GCS_IMAGES_BUCKET);
    } catch (uploadError) {
      console.error("[POST /api/gallery] upload error:", uploadError);
      return serverError("Failed to upload gallery image");
    }

    await dbConnect();

    const created = await GalleryImage.create({
      title,
      img_url: imgUrl,
      created_by: getCreatedBy(request),
    });

    return NextResponse.json(
      {
        success: true,
        message: "Gallery image added successfully",
        data: toDTO(created.toObject()),
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("[POST /api/gallery] unexpected error:", error);
    const message =
      error instanceof Error ? error.message : "Failed to add gallery image";

    return errorResponse(
      message,
      500,
      "INTERNAL_ERROR",
      request,
      userId
    );
  }
}