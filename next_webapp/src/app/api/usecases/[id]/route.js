import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

// Validation helper
function validateUseCaseUpdate(body) {
  const errors = [];

  if (body.title !== undefined) {
    if (typeof body.title !== 'string' || body.title.trim().length === 0) {
      errors.push({ field: 'title', message: 'title must be a non-empty string' });
    } else if (body.title.trim().length > 255) {
      errors.push({ field: 'title', message: 'title must be 255 characters or fewer' });
    }
  }

  if (body.description !== undefined && body.description !== null) {
    if (typeof body.description !== 'string') {
      errors.push({ field: 'description', message: 'description must be a string' });
    }
  }

  if (body.cover_img !== undefined && body.cover_img !== null) {
    if (typeof body.cover_img !== 'string') {
      errors.push({ field: 'cover_img', message: 'cover_img must be a string' });
    }
  }

  if (body.category_id !== undefined && body.category_id !== null) {
    if (!Number.isInteger(body.category_id)) {
      errors.push({ field: 'category_id', message: 'category_id must be a whole number' });
    }
  }

  return { valid: errors.length === 0, errors };
}

// GET /api/usecases/[id]
export async function GET(request, { params }) {
  try {
    const { id } = await params;

    // Validate id is a number
    if (isNaN(id)) {
      return NextResponse.json(
        { success: false, error: 'Invalid use case ID' },
        { status: 400 }
      );
    }

    const url = new URL(request.url);
    const includeContent = url.searchParams.get('include_content') === 'true';
    const fields = includeContent
      ? '*'
      : 'id, title, description, cover_img, category_id, created_by, created_at, updated_at';

    const { data, error } = await supabase
      .from('usecases')
      .select(fields)
      .eq('id', parseInt(id))
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return NextResponse.json(
          { success: false, error: 'Use case not found' },
          { status: 404 }
        );
      }
      console.error('Supabase error:', error);
      return NextResponse.json(
        { success: false, error: 'Failed to fetch use case' },
        { status: 500 }
      );
    }

    let createdBy = null;
    if (data.created_by) {
      const { data: userDetails } = await supabase
        .from('user_details')
        .select('first_name, last_name')
        .eq('user_id', data.created_by)
        .single();

      if (userDetails?.first_name || userDetails?.last_name) {
        createdBy = `${userDetails.first_name ?? ''} ${userDetails.last_name ?? ''}`.trim();
      } else {
        // fallback to email from user table
        const { data: userData } = await supabase
          .from('user')
          .select('email')
          .eq('id', data.created_by)
          .single();
        if (userData?.email) createdBy = userData.email;
      }
    }

    return NextResponse.json({
      success: true,
      data: { ...data, created_by_name: createdBy },
    });
  } catch (error) {
    console.error('Error fetching use case:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch use case' },
      { status: 500 }
    );
  }
}

// PUT /api/usecases/[id]
export async function PUT(request, { params }) {
  try {
    const { id } = await params;

    if (isNaN(id)) {
      return NextResponse.json(
        { success: false, error: 'Invalid use case ID' },
        { status: 400 }
      );
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return NextResponse.json(
        { success: false, error: 'Invalid JSON body' },
        { status: 400 }
      );
    }

    // Validate input fields
    const validation = validateUseCaseUpdate(body);
    if (!validation.valid) {
      return NextResponse.json(
        { success: false, message: 'Validation failed', errors: validation.errors },
        { status: 400 }
      );
    }

    // Build update object with only provided fields
    const updates = {};
    if (body.title !== undefined) updates.title = body.title.trim();
    if (body.description !== undefined) updates.description = body.description;
    if (body.cover_img !== undefined) updates.cover_img = body.cover_img;
    if (body.category_id !== undefined) updates.category_id = body.category_id;
    if (body.content !== undefined) updates.content = body.content;

    const hasTags = Array.isArray(body.tags);

    if (Object.keys(updates).length === 0 && !hasTags) {
      return NextResponse.json(
        { success: false, error: 'No fields provided to update' },
        { status: 400 }
      );
    }

    let data = null;

    if (Object.keys(updates).length > 0) {
      const { data: updatedRow, error } = await supabase
        .from('usecases')
        .update(updates)
        .eq('id', parseInt(id))
        .select()
        .single();

      if (error) {
        if (error.code === 'PGRST116') {
          return NextResponse.json(
            { success: false, error: 'Use case not found' },
            { status: 404 }
          );
        }
        console.error('[PUT /api/usecases] update error:', error);
        return NextResponse.json(
          { success: false, error: 'Failed to update use case' },
          { status: 500 }
        );
      }

      data = updatedRow;
    }

    if (hasTags) {
      // Remove all existing tags for this use case
      await supabase
        .from('usecase_tags')
        .delete()
        .eq('usecase_id', parseInt(id));

      // Re-insert each tag (find or create, then link)
      for (const raw of body.tags) {
        if (typeof raw !== 'string' || raw.trim().length === 0) continue;

        const name = raw.trim();
        const slug = name.toLowerCase().replace(/\s+/g, '-');

        let tag;
        const { data: insertedTag, error: tagInsertError } = await supabase
          .from('tags')
          .insert({ name, slug })
          .select('id, name, slug')
          .single();

        if (tagInsertError) {
          if (tagInsertError.code === '23505') {
            const { data: existingTag } = await supabase
              .from('tags')
              .select('id, name, slug')
              .eq('slug', slug)
              .single();
            tag = existingTag;
          } else {
            console.error('[PUT /api/usecases] tag insert error:', tagInsertError);
            continue;
          }
        } else {
          tag = insertedTag;
        }

        if (tag) {
          await supabase
            .from('usecase_tags')
            .insert({ usecase_id: parseInt(id), tag_id: tag.id });
        }
      }
    }

    return NextResponse.json({
      success: true,
      message: 'Use case updated successfully',
      data: data,
    });
  } catch (error) {
    console.error('[PUT /api/usecases] unexpected error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// DELETE /api/usecases/[id]
export async function DELETE(request, { params }) {
  try {
    const { id } = await params;

    if (isNaN(id)) {
      return NextResponse.json(
        { success: false, error: 'Invalid use case ID' },
        { status: 400 }
      );
    }

    // Check if use case exists first
    const { data: existing, error: fetchError } = await supabase
      .from('usecases')
      .select('id')
      .eq('id', parseInt(id))
      .single();

    if (fetchError || !existing) {
      return NextResponse.json(
        { success: false, error: 'Use case not found' },
        { status: 404 }
      );
    }

    // Remove associated tags first to avoid FK violation
    await supabase
      .from('usecase_tags')
      .delete()
      .eq('usecase_id', parseInt(id));

    const { error } = await supabase
      .from('usecases')
      .delete()
      .eq('id', parseInt(id));

    if (error) {
      console.error('[DELETE /api/usecases] delete error:', error);
      return NextResponse.json(
        { success: false, error: 'Failed to delete use case' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Use case deleted successfully',
    });
  } catch (error) {
    console.error('[DELETE /api/usecases] unexpected error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}
