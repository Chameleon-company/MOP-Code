/**
 * GET /api/search
 *
 * Searches use cases stored in MongoDB.
 *
 * Query parameters:
 *   q          – keyword searched against title AND description (takes priority over `title`)
 *   title      – keyword searched against title only
 *   category   – numeric legacy_id string (e.g. "3") matched against category.legacy_id,
 *                or a 24-hex MongoDB ObjectId matched against category.id
 *   tag        – tag slug matched against the embedded tags[].slug array
 *   sortBy     – field to sort by: "created_at" | "updated_at" | "title" (default: "created_at")
 *   sortOrder  – "ASC" | "DESC" (default: "DESC")
 *   page       – page number (default 1)
 *   pageSize   – results per page (default 10, max 100)
 *
 * Security note:
 *   The projection explicitly whitelists safe fields.  content_file_id (the
 *   GridFS pointer) and any other internal fields are intentionally excluded so
 *   raw notebook data is never returned to the public.
 */

import { NextResponse, type NextRequest } from 'next/server';
import mongoose from 'mongoose';
import { dbConnect } from '@/lib/dbConnect';
import { UseCase } from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';

// Fields returned in search results — content_file_id is deliberately absent.
const SEARCH_PROJECTION = {
  _id: 1,
  legacy_id: 1,
  title: 1,
  description: 1,
  cover_img: 1,
  category: 1,
  tags: 1,
  created_at: 1,
  updated_at: 1,
} as const;

export async function GET(request: NextRequest) {
  try {
    await dbConnect();

    const { searchParams } = new URL(request.url);
    const q        = searchParams.get('q')        ?? null;
    const title    = searchParams.get('title')    ?? null;
    const category = searchParams.get('category') ?? null;
    const tag      = searchParams.get('tag')      ?? null;

    // ── Sort ─────────────────────────────────────────────────────────────────
    // Whitelist valid sort fields to prevent arbitrary field injection.
    const SORT_FIELD_MAP: Record<string, string> = {
      created_at:  'created_at',
      updated_at:  'updated_at',
      title:       'title',
    };
    const rawSortBy    = searchParams.get('sortBy')    ?? 'created_at';
    const rawSortOrder = searchParams.get('sortOrder') ?? 'DESC';
    const sortField = SORT_FIELD_MAP[rawSortBy] ?? 'created_at';
    const sortDir   = rawSortOrder.toUpperCase() === 'ASC' ? 1 : -1;

    const rawPage     = parseInt(searchParams.get('page')     ?? '1',  10);
    const rawPageSize = parseInt(searchParams.get('pageSize') ?? '10', 10);
    const pageSize    = Math.min(isNaN(rawPageSize) || rawPageSize < 1 ? 10 : rawPageSize, 100);

    // ── Build filter ─────────────────────────────────────────────────────────

    const filter: mongoose.FilterQuery<typeof UseCase> = {};

    // Keyword search: q wins over title when both are present.
    if (q) {
      const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(escaped, 'i');
      filter.$or = [{ title: re }, { description: re }];
    } else if (title) {
      const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      filter.title = new RegExp(escaped, 'i');
    }

    // Category filter: accept a numeric legacy_id string or a 24-hex ObjectId.
    if (category !== null) {
      const isObjectId = /^[0-9a-fA-F]{24}$/.test(category);
      const isLegacyId = /^\d+$/.test(category);

      if (!isObjectId && !isLegacyId) {
        return errorResponse('category must be a valid integer or MongoDB ObjectId', 400, 'INVALID_CATEGORY');
      }

      if (isObjectId) {
        filter['category.id'] = new mongoose.Types.ObjectId(category);
      } else {
        // Numeric string — match against the legacy_id field (stored as string).
        filter['category.legacy_id'] = category;
      }
    }

    // Tag filter: embedded tags[].slug — no separate collection lookup needed.
    if (tag !== null) {
      filter['tags.slug'] = tag;
    }

    // ── Query ────────────────────────────────────────────────────────────────

    // Fetch all matching docs to get an accurate total for pagination.
    // For very large collections a two-step countDocuments + paginated find
    // would be more efficient, but the use-case collection is small enough
    // that a single lean pass is fine here.
    const allResults = await UseCase
      .find(filter)
      .select(SEARCH_PROJECTION)
      .sort({ [sortField]: sortDir })
      .lean();

    // Handle unknown tag slug: embedded filter returning 0 results is fine,
    // but we mirror the old behaviour of returning an empty result set rather
    // than a 404.
    const total      = allResults.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const page       = Math.min(isNaN(rawPage) || rawPage < 1 ? 1 : rawPage, totalPages);

    const paginatedResults = allResults.slice((page - 1) * pageSize, page * pageSize);

    return NextResponse.json(
      {
        success: true,
        data: {
          results: paginatedResults,
          pagination: {
            page,
            pageSize,
            total,
            totalPages,
            hasNext:  page < totalPages,
            hasPrev:  page > 1,
          },
          filters: { q, title, category, tag },
        },
      },
      { status: 200 },
    );
  } catch (error) {
    console.error('[GET /api/search] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
