import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/home/categories
// Returns all categories with name, description and cover image.
// Public — no auth required (used on the home page).
export async function GET() {
  try {
    const { data, error } = await supabase
      .from('categories')
      .select('id, category_name, description, cover_img')
      .order('category_name', { ascending: true });

    if (error) {
      console.error('[GET /api/home/categories] fetch error:', error);
      return errorResponse('Failed to fetch categories', 500, 'DB_FETCH_ERROR');
    }

    return NextResponse.json({
      success: true,
      data: data ?? [],
    });
  } catch (error) {
    console.error('[GET /api/home/categories] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
