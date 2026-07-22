import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/statistics/total-count
// Returns the total number of use cases.
// Public — no auth required.
export async function GET() {
  try {
    const { count, error } = await supabase
      .from('usecases')
      .select('*', { count: 'exact', head: true });

    if (error) {
      console.error('[GET /api/statistics/total-count] fetch error:', error);
      return errorResponse('Failed to fetch total count', 500, 'DB_FETCH_ERROR');
    }

    return NextResponse.json({ success: true, total: count });
  } catch (error) {
    console.error('[GET /api/statistics/total-count] unexpected error:', error);
    return errorResponse('Internal server error', 500);
  }
}