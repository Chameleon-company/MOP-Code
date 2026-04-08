import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

export async function POST(request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch {
      return errorResponse('Invalid JSON body', 400, 'INVALID_JSON');
    }

    const { title, description, cover_img, category_id, created_by, tags } = body;

    // Validate required fields
    if (typeof title !== 'string' || title.trim().length === 0) {
      return errorResponse('title is required', 400, 'MISSING_FIELDS');
    }
    if (created_by === undefined || created_by === null || created_by === '') {
      return errorResponse('created_by is required', 400, 'MISSING_FIELDS');
    }

    // Insert use case row
    const { data: usecaseRow, error: usecaseError } = await supabase
      .from('usecases')
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
      console.error('[POST /api/usecases] insert error:', usecaseError);
      throw usecaseError;
    }

    const resolvedTags = [];

    // Process tags if provided and non-empty
    if (Array.isArray(tags) && tags.length > 0) {
      for (const raw of tags) {
        if (typeof raw !== 'string' || raw.trim().length === 0) continue;

        const name = raw.trim();
        const slug = name.toLowerCase().replace(/\s+/g, '-');

        // Attempt to insert the tag; handle unique constraint violation (23505)
        const { data: insertedTag, error: tagInsertError } = await supabase
          .from('tags')
          .insert({ name, slug })
          .select('id, name, slug')
          .single();

        let tag;

        if (tagInsertError) {
          if (tagInsertError.code === '23505') {
            // Slug already exists — fetch the existing tag
            const { data: existingTag, error: fetchError } = await supabase
              .from('tags')
              .select('id, name, slug')
              .eq('slug', slug)
              .single();

            if (fetchError || !existingTag) {
              console.error('[POST /api/usecases] fetch existing tag error:', fetchError);
              throw fetchError ?? new Error(`Tag not found for slug: ${slug}`);
            }
            tag = existingTag;
          } else {
            console.error('[POST /api/usecases] tag insert error:', tagInsertError);
            throw tagInsertError;
          }
        } else {
          tag = insertedTag;
        }

        // Link tag to use case
        const { error: linkError } = await supabase
          .from('usecase_tags')
          .insert({ usecase_id: usecaseRow.id, tag_id: tag.id });
          

        if (linkError) {
          if (linkError.code === '23505') {
            // Link already exists — idempotent, skip silently
          } else {
            console.error('[POST /api/usecases] usecase_tags insert error:', linkError);
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
    console.error('[POST /api/usecases] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
