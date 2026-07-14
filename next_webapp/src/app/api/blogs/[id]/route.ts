import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import logger from "@/utils/logger";

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
  const blogId = Number(id);
  if (!Number.isFinite(blogId)) return badRequest("Invalid blog id");

  try {
    const { data: blog, error } = await supabase
      .from("blogs")
      .select(
        "id, created_at, updated_at, title, description, published_date, content, cover_img, created_by"
      )
      .eq("id", blogId)
      .single();

    if (error || !blog) return notFound();

    return NextResponse.json({ success: true, data: blog });
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
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const { id } = await params;
  const blogId = Number(id);
  if (!Number.isFinite(blogId)) return badRequest("Invalid blog id");

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

    const updatePayload: Record<string, any> = {
      updated_at: new Date().toISOString(),
    };

    if (title !== undefined) updatePayload.title = title;
    if (description !== undefined) updatePayload.description = description || null;
    if (publishedDate !== undefined) updatePayload.published_date = publishedDate || null;
    if (content !== undefined) updatePayload.content = content;

    // Upload new cover image if provided
    if (hasNewCover) {
      const buffer = await coverImg!.arrayBuffer();
      const ext = coverImg!.name.split(".").pop();
      const filePath = `blogs/covers/blog-cover-${userId}-${Date.now()}.${ext}`;

      const { error: uploadError } = await supabase.storage
        .from("blog-images")
        .upload(filePath, buffer, { contentType: coverImg!.type, upsert: false });

      if (uploadError) {
        console.error("[PUT /api/blogs/[id]] upload error:", uploadError);
        return serverError(`Cover image upload failed: ${uploadError.message}`);
      }

      const { data: publicUrlData } = supabase.storage
        .from("blog-images")
        .getPublicUrl(filePath);

      if (!publicUrlData?.publicUrl) return serverError("Failed to generate cover image URL");

      updatePayload.cover_img = publicUrlData.publicUrl;
    }

    const { data: blog, error } = await supabase
      .from("blogs")
      .update(updatePayload)
      .eq("id", blogId)
      .select("*")
      .single();

    if (error) {
      console.error("[PUT /api/blogs/[id]] update error:", error);
      return serverError(`Failed to update blog: ${error.message}`);
    }

    if (!blog) return notFound();

    return NextResponse.json({
      success: true,
      message: "Blog updated successfully",
      data: blog,
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
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const { id } = await params;
  const blogId = Number(id);
  if (!Number.isFinite(blogId)) return badRequest("Invalid blog id");

  try {
    // Fetch cover_img path before deleting so we can clean up storage
    const { data: existing } = await supabase
      .from("blogs")
      .select("cover_img")
      .eq("id", blogId)
      .single();

    const { error } = await supabase.from("blogs").delete().eq("id", blogId);

    if (error) {
      console.error("[DELETE /api/blogs/[id]] error:", error);
      return serverError(`Failed to delete blog: ${error.message}`);
    }

    // Best-effort: remove cover image from storage and log the deletion
    if (existing?.cover_img) {
      try {
        const imgUrl = new URL(existing.cover_img);
        const storagePath = imgUrl.pathname.split("/blog-images/")[1];
        if (storagePath) {
          await supabase.storage.from("blog-images").remove([storagePath]);
          logger.info(`Storage file deleted: blog-images/${storagePath}`, {
            source: "api",
            url: `/api/blogs/${blogId}`,
            user_id: userId,
          });
        }
      } catch {
        logger.warn(`Failed to remove storage file for blog #${blogId}`, {
          source: "api",
          url: `/api/blogs/${blogId}`,
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
