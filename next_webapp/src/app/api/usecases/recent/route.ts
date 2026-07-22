import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/usecases/recent
// Returns the 4 most recently uploaded use cases with their tags.
// Public — no auth required (used on the home page).
export async function GET() {
  try {
    const { data, error } = await supabase
      .from('usecases')
      .select(`
        id,
        title,
        description,
        cover_img,
        created_at,
        usecase_tags (
          tags (
            id,
            name,
            slug
          )
        )
      `)
      .order('created_at', { ascending: false })
      .limit(4);

    if (error) {
      console.error('[GET /api/usecases/recent] fetch error:', error);
      return errorResponse('Failed to fetch recent use cases', 500, 'DB_FETCH_ERROR');
    }

    // Flatten usecase_tags → tags for a cleaner response shape
    const usecases = (data ?? []).map(({ usecase_tags, ...rest }) => ({
      ...rest,
      tags: usecase_tags
        .map((row: { tags: { id: number; name: string; slug: string } | null }) => row.tags)
        .filter(Boolean),
    }));

    return NextResponse.json({ success: true, data: usecases });
  } catch (error) {
    console.error('[GET /api/usecases/recent] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
