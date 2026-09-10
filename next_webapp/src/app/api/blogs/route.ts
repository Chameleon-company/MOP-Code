import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Blog from "@/models/mongoose/Blog";
import User from "@/models/mongoose/User";
import { uploadImageToGCS } from "../library/uploadImageToGCS";
import { getAuthUser } from "../library/auth";
import { getImagesBucket } from "../library/gcsBucket";

// ─── Auth helpers ────────────────────────────────────────────────────────────

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
  // Loop until no more tags are stripped so nested/overlapping markup
  // (e.g. "<scr<script>ipt>") can't survive a single pass and reconstitute a tag.
  let previous: string;
  let current = html;
  do {
    previous = current;
    current = previous.replace(/<[^>]*>/g, "");
  } while (current !== previous);
  return current.replace(/&nbsp;/g, " ").trim();
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
  const { userId, isAuthenticated, isAdmin } = getAuthUser(request);
  if (!isAuthenticated || !userId) return unauthorized();
  if (!isAdmin) return forbidden();

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
    const buffer = Buffer.from(await coverImage!.arrayBuffer());
    const filename = `blogs/covers/blog-cover-${userId}-${Date.now()}.webp`;

    let coverImgUrl: string;
    try {
      coverImgUrl = await uploadImageToGCS(buffer, filename, getImagesBucket());
    } catch (uploadError) {
      console.error("[POST /api/blogs] upload error:", uploadError);
      return serverError("Cover image upload failed");
    }

    // Insert document
    await dbConnect();

    const created = await Blog.create({
      title: title!,
      description,
      published_date: publishedDate,
      content: content!,
      cover_img: coverImgUrl,
      created_by: getCreatedBy(request),
    });

    return NextResponse.json(
      { success: true, message: "Blog created successfully", data: toDTO(created.toObject()) },
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
//   created_by    Mongo user id — filter by author
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

    // Pagination
    const page = Math.max(1, parseInt(url.searchParams.get("page") ?? "1", 10) || 1);
    const pageSize = Math.min(
      100,
      Math.max(1, parseInt(url.searchParams.get("pageSize") ?? "10", 10) || 10)
    );
    const skip = (page - 1) * pageSize;

    // Validate date params
    if (dateFrom && !ISO_DATE_RE.test(dateFrom)) {
      return badRequest("date_from must be in YYYY-MM-DD format");
    }
    if (dateTo && !ISO_DATE_RE.test(dateTo)) {
      return badRequest("date_to must be in YYYY-MM-DD format");
    }
    if (createdByRaw && !mongoose.Types.ObjectId.isValid(createdByRaw)) {
      return badRequest("created_by must be a valid id");
    }

    await dbConnect();

    // Build filter
    const filter: Record<string, unknown> = {};

    const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

    if (search) {
      if (searchBy === "title") {
        filter.title = { $regex: escapeRegex(search), $options: "i" };
      } else if (searchBy === "description") {
        filter.description = { $regex: escapeRegex(search), $options: "i" };
      } else {
        filter.$or = [
          { title: { $regex: escapeRegex(search), $options: "i" } },
          { description: { $regex: escapeRegex(search), $options: "i" } },
        ];
      }
    }

    if (dateFrom || dateTo) {
      const range: Record<string, Date> = {};
      if (dateFrom) range.$gte = new Date(dateFrom);
      if (dateTo) range.$lte = new Date(dateTo);
      filter.published_date = range;
    }

    if (createdByRaw) filter.created_by = createdByRaw;

    const [rows, total] = await Promise.all([
      Blog.find(filter)
        .sort({ created_at: -1 })
        .skip(skip)
        .limit(pageSize)
        .lean(),
      Blog.countDocuments(filter),
    ]);

    // Look up first_name + last_name for all unique created_by user ids
    let nameByUserId: Record<string, string> = {};

    const userIds = [
      ...new Set(rows.map((b: any) => b.created_by).filter(Boolean).map(String)),
    ];

    if (userIds.length > 0) {
      const users = await User.find({ _id: { $in: userIds } })
        .select("profile.first_name profile.last_name")
        .lean();

      for (const u of users as any[]) {
        const name = [u.profile?.first_name, u.profile?.last_name]
          .filter(Boolean)
          .join(" ")
          .trim();
        nameByUserId[u._id.toString()] = name || "Admin";
      }
    }

    const data = rows.map((b: any) => ({
      ...toDTO(b),
      created_by_name: b.created_by ? (nameByUserId[String(b.created_by)] ?? "Admin") : "Admin",
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