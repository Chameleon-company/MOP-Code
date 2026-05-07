import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/statistics/dataset
// Returns aggregated dataset statistics across all tables.
// Public — no auth required.
export async function GET() {
  try {

    // 1. Total use cases count
    const { count: totalUsecases, error: usecasesError } = await supabase
      .from('usecases')
      .select('*', { count: 'exact', head: true });

    if (usecasesError) {
      console.error('[GET /api/statistics/dataset] usecases count error:', usecasesError);
      return errorResponse('Failed to fetch use cases count', 500, 'DB_FETCH_ERROR');
    }

    // 2. Total categories count
    const { count: totalCategories, error: categoriesError } = await supabase
      .from('categories')
      .select('*', { count: 'exact', head: true });

    if (categoriesError) {
      console.error('[GET /api/statistics/dataset] categories count error:', categoriesError);
      return errorResponse('Failed to fetch categories count', 500, 'DB_FETCH_ERROR');
    }

    // 3. Total tags count
    const { count: totalTags, error: tagsError } = await supabase
      .from('tags')
      .select('*', { count: 'exact', head: true });

    if (tagsError) {
      console.error('[GET /api/statistics/dataset] tags count error:', tagsError);
      return errorResponse('Failed to fetch tags count', 500, 'DB_FETCH_ERROR');
    }

    // 4. Uncategorized use cases count
    const { count: uncategorizedCount, error: uncategorizedError } = await supabase
      .from('usecases')
      .select('*', { count: 'exact', head: true })
      .is('category_id', null);

    if (uncategorizedError) {
      console.error('[GET /api/statistics/dataset] uncategorized count error:', uncategorizedError);
      return errorResponse('Failed to fetch uncategorized count', 500, 'DB_FETCH_ERROR');
    }

    // 5. Use cases that have at least one tag
    const { data: taggedUsecases, error: taggedError } = await supabase
      .from('usecase_tags')
      .select('usecase_id');

    if (taggedError) {
      console.error('[GET /api/statistics/dataset] tagged usecases error:', taggedError);
      return errorResponse('Failed to fetch tagged use cases', 500, 'DB_FETCH_ERROR');
    }

    const uniqueTaggedUsecases = new Set(taggedUsecases.map(row => row.usecase_id)).size;

    // 6. Return aggregated result
    return NextResponse.json({
      success: true,
      data: {
        total_usecases: totalUsecases ?? 0,
        total_categories: totalCategories ?? 0,
        total_tags: totalTags ?? 0,
        uncategorized_usecases: uncategorizedCount ?? 0,
        categorized_usecases: (totalUsecases ?? 0) - (uncategorizedCount ?? 0),
        usecases_with_tags: uniqueTaggedUsecases,
        usecases_without_tags: (totalUsecases ?? 0) - uniqueTaggedUsecases,
      },
    });

  } catch (error) {
    console.error('[GET /api/statistics/dataset] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
