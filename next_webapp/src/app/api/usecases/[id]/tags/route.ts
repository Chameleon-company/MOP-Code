import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/usecases/[id]/tags
// Fetch all tags associated with a specific use case ID
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    // Extract and parse the use case ID from route params
    const { id: rawId } = await params;
    const id = Number(rawId);

    // Validate that ID is a positive integer
    if (!Number.isInteger(id) || id <= 0) {
      return errorResponse('id must be a positive integer', 400, 'INVALID_ID');
    }

    // Step 1: Retrieve tag IDs linked to the given use case
    const { data: ucTags, error: ucError } = await supabase
      .from('usecase_tags') // junction table
      .select('tag_id')
      .eq('usecase_id', id);

    // Handle database error
    if (ucError) {
      console.error('[GET /api/usecases/[id]/tags] usecase_tags error:', ucError);
      throw ucError;
    }

    // If no tags found, return 404
    if (!ucTags || ucTags.length === 0) {
      return errorResponse('No tags found for this use case', 404, 'NOT_FOUND');
    }

    // Step 2: Extract tag IDs and fetch full tag details
    const tagIds = ucTags.map((row) => row.tag_id);

    const { data: tags, error: tagsError } = await supabase
      .from('tags')
      .select('id, name, slug') // only required fields
      .in('id', tagIds);

    // Handle database error while fetching tags
    if (tagsError) {
      console.error('[GET /api/usecases/[id]/tags] tags fetch error:', tagsError);
      throw tagsError;
    }

    // Return successful response with tag data
    return NextResponse.json({ success: true, data: tags ?? [] });
  } catch (error) {
    // Catch any unexpected errors
    console.error('[GET /api/usecases/[id]/tags] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}