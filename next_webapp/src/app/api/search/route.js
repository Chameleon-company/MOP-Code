import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const q = searchParams.get('q') ?? null;
    const title = searchParams.get('title') ?? null;
    const category = searchParams.get('category') ?? null;
    const tag = searchParams.get('tag') ?? null;

    const rawPage = parseInt(searchParams.get('page') ?? '1', 10);
    const rawPageSize = parseInt(searchParams.get('pageSize') ?? '10', 10);
    const pageSize = Math.min(isNaN(rawPageSize) || rawPageSize < 1 ? 10 : rawPageSize, 100);

    let categoryId = null;
    if (category !== null) {
      categoryId = parseInt(category, 10);
      if (isNaN(categoryId)) {
        return errorResponse('category must be a valid integer', 400, 'INVALID_CATEGORY');
      }
    }

    let results;

    if (q) {
      let titleQuery = supabase.from('usecases').select('*').ilike('title', `%${q}%`);
      let descQuery = supabase.from('usecases').select('*').ilike('description', `%${q}%`);
      if (categoryId !== null) {
        titleQuery = titleQuery.eq('category_id', categoryId);
        descQuery = descQuery.eq('category_id', categoryId);
      }

      const [{ data: byTitle, error: titleError }, { data: byDesc, error: descError }] =
        await Promise.all([titleQuery, descQuery]);

      if (titleError) throw titleError;
      if (descError) throw descError;

      const seen = new Set();
      results = [...(byTitle ?? []), ...(byDesc ?? [])].filter(({ id }) => {
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      });

    } else if (title) {
      let query = supabase.from('usecases').select('*').ilike('title', `%${title}%`);
      if (categoryId !== null) query = query.eq('category_id', categoryId);
      const { data, error } = await query;
      if (error) throw error;
      results = data ?? [];
    } else {
      let query = supabase.from('usecases').select('*');
      if (categoryId !== null) query = query.eq('category_id', categoryId);
      const { data, error } = await query;
      if (error) throw error;
      results = data ?? [];
    }

    if (tag !== null) {
      const { data: tagRow, error: tagLookupError } = await supabase
        .from('tags')
        .select('id')
        .eq('slug', tag)
        .single();

      if (tagLookupError || !tagRow) {
        return NextResponse.json(
          {
            success: true,
            data: {
              results: [],
              pagination: { page: 1, pageSize, total: 0, totalPages: 1, hasNext: false, hasPrev: false },
              filters: { q, title, category, tag },
            },
          },
          { status: 200 },
        );
      }

      const { data: tagLinks, error: tagLinksError } = await supabase
        .from('usecase_tags')
        .select('usecase_id')
        .eq('tag_id', tagRow.id);

      if (tagLinksError) throw tagLinksError;

      const taggedIds = new Set((tagLinks ?? []).map(({ usecase_id }) => usecase_id));
      results = results.filter(({ id }) => taggedIds.has(id));
    }

    const total = results.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const page = Math.min(isNaN(rawPage) || rawPage < 1 ? 1 : rawPage, totalPages);
    const paginatedResults = results.slice((page - 1) * pageSize, page * pageSize);

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
            hasNext: page < totalPages,
            hasPrev: page > 1,
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
