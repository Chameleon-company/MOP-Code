import { NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import Category from "@/models/mongoose/Category";
import Tag from "@/models/mongoose/Tag";
import User from "@/models/mongoose/User";
import { getGridFSBucket } from "@/lib/gridfs";
import { errorResponse } from "@/app/api/library/errorResponse";
import { toUseCaseDTO } from "@/app/api/library/useCaseDto";

// GridFS/streaming needs the Node runtime (mongoose, node:stream) — not edge.
export const runtime = "nodejs";

// Escape user input before it's used to build a RegExp, so keyword search
// can't inject regex metacharacters or degrade into catastrophic backtracking.
function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Server-side cap on notebook JSON size, enforced BEFORE any GridFS write —
// measured on the actual byte length (Buffer.byteLength), not .length,
// since .length counts UTF-16 code units and would under-count for any
// multi-byte characters in the notebook (cell text, unicode in outputs, etc).
const MAX_NOTEBOOK_BYTES = 60 * 1024 * 1024; // 60 MB

// Resolve an embedded category ref from an incoming category_id. The admin
// add-form still sources category_id from /api/categories, which is still
// Supabase-backed (a Postgres numeric id) — there is no live Mongo Category
// collection being written to yet. Try a real Mongo ObjectId first (forward
// -compatible with a future Mongo-backed category picker), then fall back
// to Category.legacy_id (the string field this migration already carries
// the same numeric id under — see Category.ts). If neither resolves, still
// keep the id as a legacy-only reference rather than dropping it — same
// "carry the id even without a live match" pattern used elsewhere in this
// migration (UseCase.legacy_id, TagRefSchema).
async function resolveCategoryRef(categoryId) {
  if (categoryId === undefined || categoryId === null || categoryId === "") {
    return null;
  }
  const idStr = String(categoryId);

  let category = null;
  if (mongoose.Types.ObjectId.isValid(idStr)) {
    category = await Category.findById(idStr).lean();
  }
  if (!category) {
    category = await Category.findOne({ legacy_id: idStr }).lean();
  }

  if (category) {
    return {
      id: category._id,
      legacy_id: category.legacy_id ?? idStr,
      category_name: category.category_name ?? null,
    };
  }

  return { id: null, legacy_id: idStr, category_name: null };
}

// Resolve created_by (a legacy Postgres numeric user id, read out of the
// client's localStorage session) to a Mongo User ObjectId, same
// legacy-id-fallback approach as resolveCategoryRef. Unlike category/tags,
// this field has no legacy_id sibling on the UseCase schema itself (it's a
// bare `Schema.Types.ObjectId, ref: "User"`), so an id that resolves to
// nothing is simply dropped (stored as null) rather than blocking creation
// — there's nowhere on this field to keep the raw legacy id if it doesn't
// resolve.
async function resolveCreatedBy(createdBy) {
  if (createdBy === undefined || createdBy === null || createdBy === "") {
    return null;
  }
  const idStr = String(createdBy);

  if (mongoose.Types.ObjectId.isValid(idStr)) {
    const user = await User.findById(idStr).select("_id").lean();
    if (user) return user._id;
  }

  const user = await User.findOne({ legacy_id: idStr }).select("_id").lean();
  return user ? user._id : null;
}

// Find-or-create a Tag document for each incoming tag name, then embed a
// denormalized ref on the doc — the Mongo equivalent of the old
// find-or-create-then-link-via-usecase_tags logic, minus the join table
// (tags live directly on the doc now, see TagRefSchema in UseCase.ts).
async function resolveTagRefs(tagNames) {
  if (!Array.isArray(tagNames) || tagNames.length === 0) return [];

  const refs = [];
  const seenSlugs = new Set();

  for (const raw of tagNames) {
    if (typeof raw !== "string" || raw.trim().length === 0) continue;

    const name = raw.trim();
    const slug = name.toLowerCase().replace(/\s+/g, "-");
    if (seenSlugs.has(slug)) continue;
    seenSlugs.add(slug);

    // Atomic find-or-create — replaces the old insert-then-catch-23505-
    // then-refetch dance, which Mongo's upsert makes unnecessary.
    const tag = await Tag.findOneAndUpdate(
      { slug },
      { $setOnInsert: { name, slug } },
      { upsert: true, new: true },
    ).lean();

    refs.push({
      id: tag._id,
      legacy_id: tag.legacy_id ?? null,
      name: tag.name,
      slug: tag.slug,
    });
  }

  return refs;
}

// POST /api/usecases
// Create a use case (Mongo + GridFS, ADMIN — auth enforcement is commit 6,
// unchanged here). Notebook JSON still arrives as a `content` string field
// in the body, same shape the admin add-form has always sent — it's
// written to GridFS here instead of an inline Postgres column.
export async function POST(request) {
  // Tracks a GridFS file uploaded in this request so it can be cleaned up
  // if the UseCase doc save fails afterward — never leave a doc pointing at
  // a missing file, but a harmless orphaned file (if cleanup itself fails)
  // is an acceptable worst case.
  let uploadedFileId = null;

  try {
    let body;
    try {
      body = await request.json();
    } catch (error) {
      if (error instanceof SyntaxError) {
        return errorResponse("Invalid JSON body", 400, "INVALID_JSON", request);
      }
      throw error;
    }

    const { title, description, cover_img, category_id, created_by, tags, content } =
      body;

    if (typeof title !== "string" || title.trim().length === 0) {
      return errorResponse("title is required", 400, "MISSING_FIELDS", request);
    }
    if (created_by === undefined || created_by === null || created_by === "") {
      return errorResponse("created_by is required", 400, "MISSING_FIELDS", request);
    }

    // content is optional — a use case can be created without a notebook,
    // same as before (content_file_id/content_type just stay null).
    let contentType = null;
    let notebookBuffer = null;

    if (content !== undefined && content !== null && content !== "") {
      if (typeof content !== "string") {
        return errorResponse("content must be a JSON string", 400, "INVALID_CONTENT", request);
      }

      const contentBytes = Buffer.byteLength(content, "utf8");
      if (contentBytes > MAX_NOTEBOOK_BYTES) {
        return errorResponse(
          `Notebook content exceeds the ${MAX_NOTEBOOK_BYTES / (1024 * 1024)} MB limit`,
          413,
          "CONTENT_TOO_LARGE",
          request,
        );
      }

      let parsedNotebook;
      try {
        parsedNotebook = JSON.parse(content);
      } catch (error) {
        if (error instanceof SyntaxError) {
          return errorResponse(
            "content is not valid notebook JSON",
            400,
            "INVALID_NOTEBOOK_JSON",
            request,
          );
        }
        throw error;
      }

      if (
        !parsedNotebook ||
        typeof parsedNotebook !== "object" ||
        !Array.isArray(parsedNotebook.cells)
      ) {
        return errorResponse(
          "content must be a notebook JSON object with a cells array",
          400,
          "INVALID_NOTEBOOK_FORMAT",
          request,
        );
      }

      contentType = "notebook";
      notebookBuffer = Buffer.from(content, "utf8");
    }

    await dbConnect();

    // Resolve category/tags/created_by (pure Mongo reads + tag upserts)
    // before touching GridFS — if any of these throw, nothing has been
    // uploaded yet, so there's nothing to clean up.
    const [categoryRef, createdByRef, tagRefs] = await Promise.all([
      resolveCategoryRef(category_id),
      resolveCreatedBy(created_by),
      resolveTagRefs(tags),
    ]);

    // Upload to GridFS last, right before the doc write — the smallest
    // possible window between "file exists" and "doc references it".
    if (notebookBuffer) {
      const bucket = await getGridFSBucket();
      const uploadStream = bucket.openUploadStream(`usecase-${Date.now()}.ipynb`, {
        metadata: { content_type: "notebook" },
      });
      uploadedFileId = uploadStream.id;

      await new Promise((resolve, reject) => {
        uploadStream.on("finish", resolve);
        uploadStream.on("error", reject);
        uploadStream.end(notebookBuffer);
      });
    }

    let created;
    try {
      created = await UseCase.create({
        title: title.trim(),
        description: description ?? null,
        cover_img: cover_img ?? null,
        content_file_id: uploadedFileId,
        content_type: contentType,
        category: categoryRef,
        tags: tagRefs,
        created_by: createdByRef,
      });
    } catch (saveError) {
      // Doc save failed after the file was already uploaded — clean up the
      // orphan rather than leaving it dangling forever. Best-effort: log if
      // the cleanup delete itself fails, don't let that mask saveError.
      if (uploadedFileId) {
        try {
          const bucket = await getGridFSBucket();
          await bucket.delete(uploadedFileId);
        } catch (cleanupError) {
          console.error(
            `[POST /api/usecases] failed to clean up orphaned GridFS file ${uploadedFileId} after doc save failure:`,
            cleanupError,
          );
        }
      }
      throw saveError;
    }

    return NextResponse.json(
      { success: true, data: toUseCaseDTO(created.toObject()) },
      { status: 201 },
    );
  } catch (error) {
    if (error instanceof Error && error.name === "ValidationError") {
      return errorResponse(error.message, 400, "VALIDATION_ERROR", request);
    }
    console.error("[POST /api/usecases] unexpected error:", error);
    return errorResponse("Internal server error", 500, "INTERNAL_ERROR", request);
  }
}

// GET /api/usecases
// Paginated/filterable list of use cases (PUBLIC). Mongo-backed — notebook
// bytes live in GridFS (see /api/usecases/[id]/content), never on this doc,
// so keyword search only covers title/description now, not notebook content.
export async function GET(request) {
  try {
    const url = new URL(request.url);

    const q = url.searchParams.get("q")?.trim() || "";
    const search = url.searchParams.get("search")?.trim() || "";
    const keyword = (q || search).slice(0, 200);

    const categoryId = url.searchParams.get("category_id");
    const tagId = url.searchParams.get("tag_id");
    const tagIds = url.searchParams.get("tag_ids");
    const tagSlug = url.searchParams.get("tag")?.trim();
    // tag_name: search tags by name (ilike), used by the usecases explore page
    const tagName = url.searchParams.get("tag_name")?.trim();

    // Validate page
    const rawPage = url.searchParams.get("page");
    if (rawPage !== null && (isNaN(Number(rawPage)) || Number(rawPage) < 1)) {
      return errorResponse(
        "page must be a positive number",
        400,
        "INVALID_PAGE",
        request,
      );
    }
    const page = Math.max(1, parseInt(rawPage ?? "1", 10) || 1);

    // Validate pageSize
    const rawPageSize = url.searchParams.get("pageSize");
    if (
      rawPageSize !== null &&
      (isNaN(Number(rawPageSize)) || Number(rawPageSize) < 1)
    ) {
      return errorResponse(
        "pageSize must be a positive number",
        400,
        "INVALID_PAGE_SIZE",
        request,
      );
    }
    if (rawPageSize !== null && Number(rawPageSize) > 100) {
      return errorResponse(
        "pageSize cannot exceed 100",
        400,
        "INVALID_PAGE_SIZE",
        request,
      );
    }
    const pageSize = Math.max(1, parseInt(rawPageSize ?? "10", 10) || 10);

    // Validate search_by — "content" was a valid value under the old
    // Supabase-backed route (it ilike-searched the inline `content` column).
    // Notebook content now lives in GridFS, not on this document, so it's no
    // longer searchable here.
    const searchBy = url.searchParams.get("search_by")?.trim() || "all";
    const validSearchBy = ["title", "description", "all"];
    if (!validSearchBy.includes(searchBy)) {
      return errorResponse(
        "search_by must be one of: title, description, all",
        400,
        "INVALID_SEARCH_BY",
        request,
      );
    }

    await dbConnect();

    const filter = {};

    if (keyword) {
      const regex = new RegExp(escapeRegex(keyword), "i");
      if (searchBy === "title") {
        filter.title = regex;
      } else if (searchBy === "description") {
        filter.description = regex;
      } else {
        filter.$or = [{ title: regex }, { description: regex }];
      }
    }

    // category_id carried a numeric Postgres id under the old contract. The
    // Mongo doc embeds both the new ObjectId (category.id) and the old
    // numeric id as a string (category.legacy_id) — match whichever shape
    // the caller sent.
    if (categoryId) {
      if (mongoose.Types.ObjectId.isValid(categoryId)) {
        filter["category.id"] = categoryId;
      } else {
        filter["category.legacy_id"] = categoryId;
      }
    }

    if (tagSlug) {
      filter["tags.slug"] = tagSlug;
    }

    if (tagName) {
      filter["tags.name"] = new RegExp(escapeRegex(tagName), "i");
    }

    // tag_id / tag_ids: same legacy-numeric-vs-ObjectId split as category_id.
    const tagFilterValues = [];
    if (tagId) tagFilterValues.push(tagId);
    if (tagIds) {
      tagFilterValues.push(
        ...tagIds
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      );
    }

    if (tagFilterValues.length > 0) {
      const objectIdValues = tagFilterValues.filter((v) =>
        mongoose.Types.ObjectId.isValid(v),
      );
      const legacyValues = tagFilterValues.filter(
        (v) => !mongoose.Types.ObjectId.isValid(v),
      );

      const tagOr = [];
      if (objectIdValues.length > 0) {
        tagOr.push({ "tags.id": { $in: objectIdValues } });
      }
      if (legacyValues.length > 0) {
        tagOr.push({ "tags.legacy_id": { $in: legacyValues } });
      }

      if (filter.$or) {
        // Keyword search already claimed $or — combine both conditions with $and.
        filter.$and = [{ $or: filter.$or }, { $or: tagOr }];
        delete filter.$or;
      } else {
        filter.$or = tagOr;
      }
    }

    const from = (page - 1) * pageSize;

    const [total, docs] = await Promise.all([
      UseCase.countDocuments(filter),
      UseCase.find(filter)
        .sort({ created_at: -1 })
        .skip(from)
        .limit(pageSize)
        .lean(),
    ]);

    return NextResponse.json({
      success: true,
      data: docs.map(toUseCaseDTO),
      count: docs.length,
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("[GET /api/usecases] unexpected error:", error);
    return errorResponse("Internal server error", 500, "INTERNAL_ERROR", request);
  }
}
