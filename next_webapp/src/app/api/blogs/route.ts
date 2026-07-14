import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";

// ─── Auth helpers ────────────────────────────────────────────────────────────

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
function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// ─── Validation ───────────────────────────────────────────────────────────────

const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"];
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5 MB
const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();
}

interface BlogFields {
  title?: string;
  description?: string | null;
  publishedDate?: string | null;
  content?: string;
}

function validateBlogFields(
  fields: BlogFields,
  requireAll: boolean
): Record<string, string> {
  const errors: Record<string, string> = {};

  const { title, description, publishedDate, content } = fields;

  // title
  if (requireAll && !title) {
    errors.title = "Title is required";
  } else if (title !== undefined) {
    if (title.length < 3) errors.title = "Title must be at least 3 characters";
    else if (title.length > 255) errors.title = "Title must be 255 characters or fewer";
  }

  // description (optional but bounded)
  if (description && description.length > 500) {
    errors.description = "Description must be 500 characters or fewer";
  }

  // published_date
  if (requireAll && !publishedDate) {
    errors.published_date = "Published date is required";
  } else if (publishedDate && !ISO_DATE_RE.test(publishedDate)) {
    errors.published_date = "Published date must be in YYYY-MM-DD format";
  }

  // content
  if (requireAll && !content) {
    errors.content = "Content is required";
  } else if (content !== undefined && !stripHtml(content)) {
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

// ─── POST — create blog ───────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  try {
    const formData = await request.formData();

    const title = formData.get("title")?.toString().trim();
    const description = formData.get("description")?.toString().trim() || null;
    const publishedDate = formData.get("published_date")?.toString().trim() || null;
    const content = formData.get("content")?.toString().trim();
    const coverImage = formData.get("cover_img") as File | null;

    // Field validation
    const fieldErrors = validateBlogFields(
      { title, description, publishedDate, content },
      true
    );
    if (!coverImage || coverImage.size === 0) {
      fieldErrors.cover_img = "Cover image is required";
    } else {
      const imgError = validateImage(coverImage);
      if (imgError) fieldErrors.cover_img = imgError;
    }

    if (Object.keys(fieldErrors).length > 0) {
      return badRequest("Validation failed", fieldErrors);
    }

    // Upload cover image
    const buffer = await coverImage!.arrayBuffer();
    const ext = coverImage!.name.split(".").pop();
    const filePath = `blogs/covers/blog-cover-${userId}-${Date.now()}.${ext}`;

    const { error: uploadError } = await supabase.storage
      .from("blog-images")
      .upload(filePath, buffer, { contentType: coverImage!.type, upsert: false });

    if (uploadError) {
      console.error("[POST /api/blogs] upload error:", uploadError);
      return serverError(`Cover image upload failed: ${uploadError.message}`);
    }

    const { data: publicUrlData } = supabase.storage
      .from("blog-images")
      .getPublicUrl(filePath);

    if (!publicUrlData?.publicUrl) return serverError("Failed to generate cover image URL");

    // Insert row
    const { data: blog, error: insertError } = await supabase
      .from("blogs")
      .insert({
        title: title!,
        description,
        published_date: publishedDate,
        content: content!,
        cover_img: publicUrlData.publicUrl,
        created_by: userId,
        updated_at: new Date().toISOString(),
      })
      .select("*")
      .single();

    if (insertError) {
      console.error("[POST /api/blogs] insert error:", insertError);
      return serverError(`Failed to create blog: ${insertError.message}`);
    }

    return NextResponse.json(
      { success: true, message: "Blog created successfully", data: blog },
      { status: 201 }
    );
  } catch (error) {
    console.error("[POST /api/blogs] error:", error);
    return serverError("Failed to create blog");
  }
}

// ─── GET — list blogs ─────────────────────────────────────────────────────────
//
// Query params:
//   search        keyword — searched against title, description, or both
//   search_by     "title" | "description" | "all" (default: "all")
//   date_from     YYYY-MM-DD  filter published_date >=
//   date_to       YYYY-MM-DD  filter published_date <=
//   created_by    number      filter by author user id
//   page          number (default 1)
//   pageSize      number (default 10)

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);

    // Search
    const search = url.searchParams.get("search")?.trim() || "";
    const searchBy = url.searchParams.get("search_by")?.trim() || "all";

    // Date range
    const dateFrom = url.searchParams.get("date_from")?.trim() || "";
    const dateTo = url.searchParams.get("date_to")?.trim() || "";

    // Author filter
    const createdByRaw = url.searchParams.get("created_by")?.trim();
    const createdBy = createdByRaw ? Number(createdByRaw) : null;

    // Pagination
    const page = Math.max(1, parseInt(url.searchParams.get("page") ?? "1", 10) || 1);
    const pageSize = Math.min(
      100,
      Math.max(1, parseInt(url.searchParams.get("pageSize") ?? "10", 10) || 10)
    );
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;

    // Validate date params
    if (dateFrom && !ISO_DATE_RE.test(dateFrom)) {
      return badRequest("date_from must be in YYYY-MM-DD format");
    }
    if (dateTo && !ISO_DATE_RE.test(dateTo)) {
      return badRequest("date_to must be in YYYY-MM-DD format");
    }
    if (createdByRaw && (createdBy === null || !Number.isFinite(createdBy))) {
      return badRequest("created_by must be a valid number");
    }

    let query = supabase
      .from("blogs")
      .select(
        "id, created_at, updated_at, title, description, published_date, content, cover_img, created_by",
        { count: "exact" }
      )
      .order("created_at", { ascending: false });

    // Keyword search
    if (search) {
      if (searchBy === "title") {
        query = query.ilike("title", `%${search}%`);
      } else if (searchBy === "description") {
        query = query.ilike("description", `%${search}%`);
      } else {
        query = query.or(`title.ilike.%${search}%,description.ilike.%${search}%`);
      }
    }

    // Date range filter
    if (dateFrom) query = query.gte("published_date", dateFrom);
    if (dateTo) query = query.lte("published_date", dateTo);

    // Author filter
    if (createdBy) query = query.eq("created_by", createdBy);

    const { data: rows, error, count } = await query.range(from, to);

    if (error) {
      console.error("[GET /api/blogs] error:", error);
      return serverError(`Failed to fetch blogs: ${error.message}`);
    }

    const total = count ?? 0;
    const blogRows = rows ?? [];

    // Step 2: look up first_name + last_name for all unique created_by user IDs
    let nameByUserId: Record<number, string> = {};

    if (blogRows.length > 0) {
      const userIds = [...new Set(blogRows.map((b) => b.created_by).filter(Boolean))];

      const { data: details } = await supabase
        .from("user_details")
        .select("user_id, first_name, last_name")
        .in("user_id", userIds);

      if (details) {
        for (const d of details) {
          const name = [d.first_name, d.last_name].filter(Boolean).join(" ").trim();
          nameByUserId[d.user_id] = name || "Admin";
        }
      }
    }

    const data = blogRows.map((b) => ({
      ...b,
      created_by_name: nameByUserId[b.created_by] ?? "Admin",
    }));

    return NextResponse.json({
      success: true,
      data,
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("[GET /api/blogs] error:", error);
    return serverError("Failed to fetch blogs");
  }
}
