import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Blog from "@/models/mongoose/Blog";
import { supabase } from "@/library/supabaseClient";
import { uploadImageToGCS } from "../../library/uploadImageToGCS";
import logger from "@/utils/logger";

const GCS_IMAGES_BUCKET = process.env.GCS_IMAGES_BUCKET ?? "mop-images";

// ─── Auth helpers ─────────────────────────────────────────────────────────────

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

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}

// ─── Response helpers ─────────────────────────────────────────────────────────

function unauthorized() {
  return NextResponse.json({ success: false, message: "Unauthorised" }, { status: 401 });
}
function forbidden() {
  return NextResponse.json({ success: false, message: "Forbidden - Admin only" }, { status: 403 });
}
function badRequest(message: string, errors?: Record<string, string>) {
  return NextResponse.json({ success: false, message, ...(errors && { errors }) }, { status: 400 });
}
function notFound(message = "Blog not found") {
  return NextResponse.json({ success: false, message }, { status: 404 });
}
function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// ─── Validation ───────────────────────────────────────────────────────────────

const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"];
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();
}

function validateUpdateFields(fields: {
  title?: string;
  description?: string;
  publishedDate?: string;
  content?: string;
}): Record<string, string> {
  const errors: Record<string, string> = {};
  const { title, description, publishedDate, content } = fields;

  if (title !== undefined) {
    if (title.length < 3) errors.title = "Title must be at least 3 characters";
    else if (title.length > 255) errors.title = "Title must be 255 characters or fewer";
  }

  if (description !== undefined && description.length > 500) {
    errors.description = "Description must be 500 characters or fewer";
  }

  if (publishedDate !== undefined && publishedDate !== "" && !ISO_DATE_RE.test(publishedDate)) {
    errors.published_date = "Published date must be in YYYY-MM-DD format";
  }

  if (content !== undefined && !stripHtml(content)) {
    errors.content = "Content cannot be empty";
  }

  return errors;
}

function validateImage(file: File): string | null {
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    return "Cover image must be JPEG, PNG, GIF, or WebP";
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return "Cover image must be smaller than 5 MB";
  }
  return null;
}

// ─── GET — single blog ────────────────────────────────────────────────────────

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  if (!mongoose.Types.ObjectId.isValid(id)) return badRequest("Invalid blog id");

  try {
    await dbConnect();

    const blog = await Blog.findById(id).lean();
    if (!blog) return notFound();

    return NextResponse.json({ success: true, data: toDTO(blog) });
  } catch (error) {
    console.error("[GET /api/blogs/[id]] error:", error);
    return serverError("Failed to fetch blog");
  }
}

// ─── PUT — update blog ────────────────────────────────────────────────────────

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { userId, isAuthenticated, isAdmin } = getAuthUser(request);
  if (!isAuthenticated || !userId) return unauthorized();
  if (!isAdmin) return forbidden();

  const { id } = await params;
  if (!mongoose.Types.ObjectId.isValid(id)) return badRequest("Invalid blog id");

  try {
    const formData = await request.formData();

    const title = formData.get("title")?.toString().trim();
    const description = formData.get("description")?.toString().trim();
    const publishedDate = formData.get("published_date")?.toString().trim();
    const content = formData.get("content")?.toString().trim();
    const coverImg = formData.get("cover_img") as File | null;
    const hasNewCover = coverImg && coverImg.size > 0;

    if (!title && !description && !publishedDate && !content && !hasNewCover) {
      return badRequest("At least one field is required to update");
    }

    // Field validation (partial — only validate fields that were provided)
    const fieldErrors = validateUpdateFields({ title, description, publishedDate, content });
    if (hasNewCover) {
      const imgError = validateImage(coverImg!);
      if (imgError) fieldErrors.cover_img = imgError;
    }

    if (Object.keys(fieldErrors).length > 0) {
      return badRequest("Validation failed", fieldErrors);
    }

    await dbConnect();

    const existing = await Blog.findById(id);
    if (!existing) return notFound();

    const updateFields: Record<string, unknown> = {};
    if (title !== undefined) updateFields.title = title;
    if (description !== undefined) updateFields.description = description || null;
    if (publishedDate !== undefined) {
      updateFields.published_date = publishedDate ? new Date(publishedDate) : null;
    }
    if (content !== undefined) updateFields.content = content;
    existing.set(updateFields);

    // Upload new cover image if provided
    if (hasNewCover) {
      const buffer = Buffer.from(await coverImg!.arrayBuffer());
      const filename = `blogs/covers/blog-cover-${userId}-${Date.now()}.webp`;

      try {
        existing.cover_img = await uploadImageToGCS(buffer, filename, GCS_IMAGES_BUCKET);
      } catch (uploadError) {
        console.error("[PUT /api/blogs/[id]] upload error:", uploadError);
        return serverError("Cover image upload failed");
      }
    }

    await existing.save();

    return NextResponse.json({
      success: true,
      message: "Blog updated successfully",
      data: toDTO(existing.toObject()),
    });
  } catch (error) {
    console.error("[PUT /api/blogs/[id]] error:", error);
    return serverError("Failed to update blog");
  }
}

// ─── DELETE — remove blog ─────────────────────────────────────────────────────

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { userId, isAuthenticated, isAdmin } = getAuthUser(request);
  if (!isAuthenticated || !userId) return unauthorized();
  if (!isAdmin) return forbidden();

  const { id } = await params;
  if (!mongoose.Types.ObjectId.isValid(id)) return badRequest("Invalid blog id");

  try {
    await dbConnect();

    // Fetch cover_img path before deleting so we can clean up storage
    const existing = await Blog.findById(id).select("cover_img").lean();

    const deleted = await Blog.findByIdAndDelete(id);
    if (!deleted) return notFound();

    // Best-effort: remove cover image from storage and log the deletion
    if (existing?.cover_img) {
      try {
        const imgUrl = new URL(existing.cover_img);
        const storagePath = imgUrl.pathname.split("/blog-images/")[1];
        if (storagePath) {
          await supabase.storage.from("blog-images").remove([storagePath]);
          logger.info(`Storage file deleted: blog-images/${storagePath}`, {
            source: "api",
            url: `/api/blogs/${id}`,
            user_id: userId,
          });
        }
      } catch {
        logger.warn(`Failed to remove storage file for blog #${id}`, {
          source: "api",
          url: `/api/blogs/${id}`,
          user_id: userId,
        });
      }
    }

    return NextResponse.json({ success: true, message: "Blog deleted successfully" });
  } catch (error) {
    console.error("[DELETE /api/blogs/[id]] error:", error);
    return serverError("Failed to delete blog");
  }
}