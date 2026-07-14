import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/statistics/by-category
// Returns use case counts grouped by category.
// Public — no auth required.
export async function GET() {
  try {
    // Fetch all use cases' category_ids
    const { data: usecases, error: usecasesError } = await supabase
      .from('usecases')
      .select('category_id');

    if (usecasesError) {
      console.error('[GET /api/statistics/by-category] usecases fetch error:', usecasesError);
      return errorResponse('Failed to fetch use cases', 500, 'DB_FETCH_ERROR');
    }

    // Count by category_id
    const countMap = {};
    usecases.forEach(usecase => {
      const catId = usecase.category_id || 'uncategorized';
      countMap[catId] = (countMap[catId] || 0) + 1;
    });

    // Fetch category names
    const { data: categories, error: categoriesError } = await supabase
      .from('categories')
      .select('id, category_name');

    if (categoriesError) {
      console.error('[GET /api/statistics/by-category] categories fetch error:', categoriesError);
      return errorResponse('Failed to fetch categories', 500, 'DB_FETCH_ERROR');
    }

    // Create id to name map
    const categoryMap = {};
    categories.forEach(cat => {
      categoryMap[cat.id] = cat.category_name;
    });

    // Build result array
    const result = Object.entries(countMap).map(([catId, count]) => ({
      category: catId === 'uncategorized' ? 'Uncategorized' : (categoryMap[catId] || 'Unknown'),
      count
    }));

    // Sort by count descending
    result.sort((a, b) => b.count - a.count);

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    console.error('[GET /api/statistics/by-category] unexpected error:', error);
    return errorResponse('Internal server error', 500);
  }
}