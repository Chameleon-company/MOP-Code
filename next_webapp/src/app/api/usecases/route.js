import { NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import { getAuthUser } from "@/app/api/library/auth";
import { errorResponse } from "@/app/api/library/errorResponse";
import { toUseCaseDTO } from "@/app/api/library/useCaseDto";
import {
  resolveCategoryRef,
  resolveCreatedBy,
  resolveTagRefs,
  validateNotebookContent,
  uploadNotebookToGridFS,
  tryDeleteGridFSFile,
} from "./_shared";

// GridFS/streaming needs the Node runtime (mongoose, node:stream) — not edge.
export const runtime = "nodejs";

// Escape user input before it's used to build a RegExp, so keyword search
// can't inject regex metacharacters or degrade into catastrophic backtracking.
function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// POST /api/usecases
// Create a use case (Mongo + GridFS, ADMIN ONLY). Notebook JSON still
// arrives as a `content` string field in the body, same shape the admin
// add-form has always sent — it's written to GridFS here instead of an
// inline Postgres column.
export async function POST(request) {
  // Tracks a GridFS file uploaded in this request so it can be cleaned up
  // if the UseCase doc save fails afterward — never leave a doc pointing at
  // a missing file, but a harmless orphaned file (if cleanup itself fails)
  // is an acceptable worst case.
  let uploadedFileId = null;

  try {
    const { userId, isAuthenticated, isAdmin } = getAuthUser(request);
    if (!isAuthenticated) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request);
    }
    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request);
    }

    let body;
    try {
      body = await request.json();
    } catch (error) {
      if (error instanceof SyntaxError) {
        return errorResponse("Invalid JSON body", 400, "INVALID_JSON", request);
      }
      throw error;
    }

    // created_by is intentionally NOT read from the body it's stamped from
    // the authenticated admin's own id below. A client-supplied creator id
    // is never trusted (would let one admin attribute a use case to anyone).
    const { title, description, cover_img, category_id, tags, content } = body;

    if (typeof title !== "string" || title.trim().length === 0) {
      return errorResponse("title is required", 400, "MISSING_FIELDS", request);
    }

    // content is optional — a use case can be created without a notebook,
    // same as before (content_file_id/content_type just stay null).
    let contentType = null;
    let notebookBuffer = null;

    if (content !== undefined && content !== null && content !== "") {
      const validation = validateNotebookContent(content);
      if (!validation.valid) {
        return errorResponse(validation.message, validation.status, validation.code, request);
      }
      contentType = validation.contentType;
      notebookBuffer = validation.notebookBuffer;
    }

    await dbConnect();

    // Resolve category/tags/created_by (pure Mongo reads + tag upserts)
    // before touching GridFS if any of these throw, nothing has been
    // uploaded yet, so there's nothing to clean up.
    const [categoryRef, createdByRef, tagRefs] = await Promise.all([
      resolveCategoryRef(category_id),
      resolveCreatedBy(userId),
      resolveTagRefs(tags),
    ]);

    // Upload to GridFS last, right before the doc write — the smallest
    // possible window between "file exists" and "doc references it".
    if (notebookBuffer) {
      uploadedFileId = await uploadNotebookToGridFS(notebookBuffer);
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
      // orphan rather than leaving it dangling forever.
      await tryDeleteGridFSFile(uploadedFileId, "orphaned upload after failed create");
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
