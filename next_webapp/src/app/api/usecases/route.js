import { NextResponse } from "next/server";
import mongoose from "mongoose";
import { supabase } from "@/library/supabaseClient";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import { errorResponse } from "@/app/api/library/errorResponse";
import { toUseCaseDTO } from "@/app/api/library/useCaseDto";

// Escape user input before it's used to build a RegExp, so keyword search
// can't inject regex metacharacters or degrade into catastrophic backtracking.
function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// POST is unchanged from the pre-GridFS version — still Supabase-backed.
// Migrating create to Mongo/GridFS is a later commit.
export async function POST(request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch {
      return errorResponse("Invalid JSON body", 400, "INVALID_JSON");
    }

    const { title, description, cover_img, category_id, created_by, tags, content } =
      body;

    if (typeof title !== "string" || title.trim().length === 0) {
      return errorResponse("title is required", 400, "MISSING_FIELDS");
    }
    if (created_by === undefined || created_by === null || created_by === "") {
      return errorResponse("created_by is required", 400, "MISSING_FIELDS");
    }

    const { data: usecaseRow, error: usecaseError } = await supabase
      .from("usecases")
      .insert({
        title: title.trim(),
        description: description ?? null,
        cover_img: cover_img ?? null,
        category_id: category_id ?? null,
        created_by,
        content: content ?? null,
      })
      .select()
      .single();

    if (usecaseError) {
      console.error("[POST /api/usecases] insert error:", usecaseError);
      throw usecaseError;
    }

    const resolvedTags = [];

    if (Array.isArray(tags) && tags.length > 0) {
      for (const raw of tags) {
        if (typeof raw !== "string" || raw.trim().length === 0) continue;

        const name = raw.trim();
        const slug = name.toLowerCase().replace(/\s+/g, "-");

        const { data: insertedTag, error: tagInsertError } = await supabase
          .from("tags")
          .insert({ name, slug })
          .select("id, name, slug")
          .single();

        let tag;

        if (tagInsertError) {
          if (tagInsertError.code === "23505") {
            const { data: existingTag, error: fetchError } = await supabase
              .from("tags")
              .select("id, name, slug")
              .eq("slug", slug)
              .single();

            if (fetchError || !existingTag) {
              console.error(
                "[POST /api/usecases] fetch existing tag error:",
                fetchError,
              );
              throw fetchError ?? new Error(`Tag not found for slug: ${slug}`);
            }
            tag = existingTag;
          } else {
            console.error(
              "[POST /api/usecases] tag insert error:",
              tagInsertError,
            );
            throw tagInsertError;
          }
        } else {
          tag = insertedTag;
        }

        const { error: linkError } = await supabase
          .from("usecase_tags")
          .insert({ usecase_id: usecaseRow.id, tag_id: tag.id });

        if (linkError) {
          if (linkError.code === "23505") {
            // Link already exists — idempotent, skip silently
          } else {
            console.error(
              "[POST /api/usecases] usecase_tags insert error:",
              linkError,
            );
            throw linkError;
          }
        }

        resolvedTags.push(tag);
      }
    }

    const uniqueTags = resolvedTags.filter(
      (tag, index, arr) => arr.findIndex((t) => t.id === tag.id) === index,
    );

    return NextResponse.json(
      { success: true, data: { ...usecaseRow, tags: uniqueTags } },
      { status: 201 },
    );
  } catch (error) {
    console.error("[POST /api/usecases] unexpected error:", error);
    return errorResponse("Internal server error", 500, "INTERNAL_ERROR");
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
