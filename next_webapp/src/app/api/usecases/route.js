import { NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import { errorResponse } from "@/app/api/library/errorResponse";

export async function POST(request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch {
      return errorResponse("Invalid JSON body", 400, "INVALID_JSON");
    }

    const { title, description, cover_img, category_id, created_by, tags } =
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

export async function GET(request) {
  try {
    const url = new URL(request.url);

    const q = url.searchParams.get("q")?.trim() || "";
    const search = url.searchParams.get("search")?.trim() || "";
    const keyword = q || search;

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
      );
    }
    if (rawPageSize !== null && Number(rawPageSize) > 100) {
      return errorResponse(
        "pageSize cannot exceed 100",
        400,
        "INVALID_PAGE_SIZE",
      );
    }
    const pageSize = Math.max(1, parseInt(rawPageSize ?? "10", 10) || 10);

    // Validate search_by
    const searchBy = url.searchParams.get("search_by")?.trim() || "all";
    const validSearchBy = ["title", "description", "all"];
    if (!validSearchBy.includes(searchBy)) {
      return errorResponse(
        "search_by must be one of: title, description, all",
        400,
        "INVALID_SEARCH_BY",
      );
    }
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;

    // Step 1: query use cases with count (no relational select — keeps count reliable)
    let query = supabase
      .from("usecases")
      .select("id, title, description, cover_img, category_id, created_at", {
        count: "exact",
      })
      .order("created_at", { ascending: false });

    // Keyword search — narrows to title, description, or both based on search_by
    if (keyword) {
      if (searchBy === "title") {
        query = query.ilike("title", `%${keyword}%`);
      } else if (searchBy === "description") {
        query = query.ilike("description", `%${keyword}%`);
      } else {
        query = query.or(
          `title.ilike.%${keyword}%,description.ilike.%${keyword}%`,
        );
      }
    }

    if (categoryId) {
      const parsedCategoryId = Number(categoryId);

      if (Number.isNaN(parsedCategoryId)) {
        return errorResponse(
          "category_id must be a valid number",
          400,
          "INVALID_CATEGORY_ID",
        );
      }

      query = query.eq("category_id", parsedCategoryId);
    }

    const tagFilterIds = [];

    if (tagId) {
      const parsedTagId = Number(tagId);

      if (Number.isNaN(parsedTagId)) {
        return errorResponse(
          "tag_id must be a valid number",
          400,
          "INVALID_TAG_ID",
        );
      }

      tagFilterIds.push(parsedTagId);
    }

    if (tagIds) {
      const parsedTagIds = tagIds
        .split(",")
        .map((id) => Number(id.trim()))
        .filter((id) => !Number.isNaN(id));

      if (parsedTagIds.length === 0) {
        return errorResponse(
          "tag_ids must contain valid numbers",
          400,
          "INVALID_TAG_IDS",
        );
      }

      tagFilterIds.push(...parsedTagIds);
    }

    if (tagSlug) {
      const { data: tagData, error: tagError } = await supabase
        .from("tags")
        .select("id")
        .eq("slug", tagSlug)
        .single();

      if (tagError || !tagData) {
        return NextResponse.json({
          success: true,
          data: [],
          count: 0,
          pagination: { page, pageSize, total: 0, totalPages: 0 },
        });
      }

      tagFilterIds.push(tagData.id);
    }

    // tag_name: look up all tags whose name matches (partial, case-insensitive)
    if (tagName) {
      const { data: matchingTags, error: tagNameError } = await supabase
        .from("tags")
        .select("id")
        .ilike("name", `%${tagName}%`);

      if (tagNameError) {
        console.error(
          "[GET /api/usecases] tag_name lookup error:",
          tagNameError,
        );
        return errorResponse("Failed to search tags", 500, "TAG_SEARCH_ERROR");
      }

      if (!matchingTags || matchingTags.length === 0) {
        return NextResponse.json({
          success: true,
          data: [],
          count: 0,
          pagination: { page, pageSize, total: 0, totalPages: 0 },
        });
      }

      tagFilterIds.push(...matchingTags.map((t) => t.id));
    }

    const uniqueTagFilterIds = [...new Set(tagFilterIds)];

    if (uniqueTagFilterIds.length > 0) {
      const { data: usecaseTags, error: tagFilterError } = await supabase
        .from("usecase_tags")
        .select("usecase_id")
        .in("tag_id", uniqueTagFilterIds);

      if (tagFilterError) {
        console.error("[GET /api/usecases] tag filter error:", tagFilterError);
        return errorResponse(
          "Failed to filter use cases by tags",
          500,
          "TAG_FILTER_ERROR",
        );
      }

      const usecaseIds = [
        ...new Set((usecaseTags || []).map((item) => item.usecase_id)),
      ];

      if (usecaseIds.length === 0) {
        return NextResponse.json({
          success: true,
          data: [],
          count: 0,
          pagination: { page, pageSize, total: 0, totalPages: 0 },
        });
      }

      query = query.in("id", usecaseIds);
    }

    // Step 1: fetch use cases with count
    const { data, error, count } = await query.range(from, to);

    if (error) {
      console.error("[GET /api/usecases] fetch error:", error);
      return NextResponse.json(
        { success: false, error: "Failed to fetch use cases" },
        { status: 500 },
      );
    }

    const total = count ?? 0;
    const rows = data || [];

    // Step 2: fetch tags for the returned use case IDs in a separate query
    let tagsByUsecaseId = {};

    if (rows.length > 0) {
      const ids = rows.map((u) => u.id);

      const { data: ucTags, error: tagsError } = await supabase
        .from("usecase_tags")
        .select("usecase_id, tags(id, name, slug)")
        .in("usecase_id", ids);

      if (!tagsError && ucTags) {
        for (const row of ucTags) {
          if (!tagsByUsecaseId[row.usecase_id])
            tagsByUsecaseId[row.usecase_id] = [];
          if (row.tags) tagsByUsecaseId[row.usecase_id].push(row.tags);
        }
      }
    }

    const usecases = rows.map((u) => ({
      ...u,
      tags: tagsByUsecaseId[u.id] || [],
    }));

    return NextResponse.json({
      success: true,
      data: usecases,
      count: usecases.length,
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("Error fetching use cases:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch use cases" },
      { status: 500 },
    );
  }
}
