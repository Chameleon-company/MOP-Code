import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import { uploadImageToStorage } from "../../library/uploadImageToStorage";
import logger from "@/utils/logger";

// ── Constants ──────────────────────────────────────────────────────────────
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5 MB
const MAX_TITLE_LENGTH = 200;

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

function notFound(message = "Gallery image not found") {
  return NextResponse.json({ success: false, message }, { status: 404 });
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// ── Shared param parsing ───────────────────────────────────────────────────
async function parseId(
  params: Promise<{ id: string }>
): Promise<number | null> {
  const { id } = await params;
  const n = Number(id);
  return Number.isFinite(n) && n > 0 ? n : null;
}

// ── GET /api/gallery/[id] ──────────────────────────────────────────────────
// Fetch a single gallery image. Requires authentication (any role).
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!getUserId(request)) return unauthorized();

  const galleryImageId = await parseId(params);
  if (!galleryImageId) return badRequest("Invalid gallery image id");

  try {
    const { data, error } = await supabase
      .from("gallery_images")
      .select("id, title, img_url, created_at, created_by")
      .eq("id", galleryImageId)
      .single();

    if (error?.code === "PGRST116" || !data) return notFound();

    if (error) {
      console.error("[GET /api/gallery/[id]] error:", error);
      return serverError("Failed to fetch gallery image");
    }

    return NextResponse.json({ success: true, data });
  } catch (error) {
    console.error("[GET /api/gallery/[id]] unexpected error:", error);
    return serverError("Failed to fetch gallery image");
  }
}

// ── PUT /api/gallery/[id] ──────────────────────────────────────────────────
// Admin only. Accepts multipart/form-data: title? (string), image? (File).
// At least one field must be provided.
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const galleryImageId = await parseId(params);
  if (!galleryImageId) return badRequest("Invalid gallery image id");

  try {
    // Verify the record exists first
    const { data: existing, error: fetchError } = await supabase
      .from("gallery_images")
      .select("id")
      .eq("id", galleryImageId)
      .single();

    if (fetchError?.code === "PGRST116" || !existing) return notFound();

    const formData = await request.formData();
    const title = formData.get("title")?.toString().trim();
    const image = formData.get("image") as File | null;

    const hasImage = image && image.size > 0;

    if (!title && !hasImage) {
      return badRequest("At least one field (title or image) is required");
    }

    const errors: Record<string, string> = {};

    if (title !== undefined) {
      if (title.length === 0) {
        errors.title = "Title cannot be empty";
      } else if (title.length > MAX_TITLE_LENGTH) {
        errors.title = `Title must be ${MAX_TITLE_LENGTH} characters or fewer`;
      }
    }

    if (hasImage) {
      if (!ALLOWED_TYPES.includes((image as File).type)) {
        errors.image = "Only JPEG, PNG, or WebP images are allowed";
      } else if ((image as File).size > MAX_IMAGE_SIZE) {
        errors.image = "Image must be under 5 MB";
      }
    }

    if (Object.keys(errors).length > 0) {
      return badRequest("Validation failed", errors);
    }

    const updatePayload: { title?: string; img_url?: string; updated_at: string } = {
      updated_at: new Date().toISOString(),
    };

    if (title) updatePayload.title = title;

    if (hasImage) {
      const uploaded = await uploadImageToStorage({
        file: image as File,
        bucket: "gallery-images",
        folder: "gallery",
        prefix: "gallery",
        userId,
      });
      updatePayload.img_url = uploaded.publicUrl;
    }

    const { data: galleryImage, error } = await supabase
      .from("gallery_images")
      .update(updatePayload)
      .eq("id", galleryImageId)
      .select("id, title, img_url, created_at, created_by")
      .single();

    if (error) {
      console.error("[PUT /api/gallery/[id]] update error:", error);
      return serverError("Failed to update gallery image");
    }

    if (!galleryImage) return notFound();

    return NextResponse.json({
      success: true,
      message: "Gallery image updated successfully",
      data: galleryImage,
    });
  } catch (error) {
    console.error("[PUT /api/gallery/[id]] unexpected error:", error);
    const message =
      error instanceof Error ? error.message : "Failed to update gallery image";
    return serverError(message);
  }
}

// ── DELETE /api/gallery/[id] ───────────────────────────────────────────────
// Admin only. Permanently removes the record (storage file is NOT deleted
// automatically — Supabase Storage cleanup can be handled separately or
// via a storage lifecycle policy).
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const galleryImageId = await parseId(params);
  if (!galleryImageId) return badRequest("Invalid gallery image id");

  try {
    const { data: existing, error: fetchError } = await supabase
      .from("gallery_images")
      .select("id, img_url")
      .eq("id", galleryImageId)
      .single();

    if (fetchError?.code === "PGRST116" || !existing) return notFound();

    const { error } = await supabase
      .from("gallery_images")
      .delete()
      .eq("id", galleryImageId);

    if (error) {
      console.error("[DELETE /api/gallery/[id]] delete error:", error);
      return serverError("Failed to delete gallery image");
    }

    // Best-effort: remove image file from storage and log the deletion
    if (existing?.img_url) {
      try {
        const imgUrl = new URL(existing.img_url);
        const storagePath = imgUrl.pathname.split("/gallery-images/")[1];
        if (storagePath) {
          await supabase.storage.from("gallery-images").remove([storagePath]);
          logger.info(`Storage file deleted: gallery-images/${storagePath}`, {
            source: "api",
            url: `/api/gallery/${galleryImageId}`,
            user_id: userId,
          });
        }
      } catch {
        logger.warn(`Failed to remove storage file for gallery image #${galleryImageId}`, {
          source: "api",
          url: `/api/gallery/${galleryImageId}`,
          user_id: userId,
        });
      }
    }

    return NextResponse.json({
      success: true,
      message: "Gallery image deleted successfully",
    });
  } catch (error) {
    console.error("[DELETE /api/gallery/[id]] unexpected error:", error);
    return serverError("Failed to delete gallery image");
  }
}
