import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import mongoose from 'mongoose';
import dbConnect from '@/lib/dbConnect';
import UseCase from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';
import { toUseCaseDTO } from '@/app/api/library/useCaseDto';
import {
  resolveCategoryRef,
  resolveCreatedBy,
  resolveTagRefs,
  validateNotebookContent,
  uploadNotebookToGridFS,
  tryDeleteGridFSFile,
} from '../_shared';

// GridFS/streaming needs the Node runtime (mongoose, node:stream) — not edge.
export const runtime = 'nodejs';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

// Basic field validation for PUT — mirrors the old Supabase-backed PUT's
// validateUseCaseUpdate, minus the category_id integer check (category_id
// can now be either a Mongo ObjectId string or a legacy Postgres numeric
// id — see resolveCategoryRef in ../_shared — so it's not validated as a
// type here, matching how POST already treats it just as permissively).
function validateBasicFields(body) {
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

  return { valid: errors.length === 0, errors };
}

// GET /api/usecases/[id]
// Mongo-backed (PUBLIC). Metadata only — content_type and content_file_id
// are returned as plain doc fields (never the notebook bytes themselves);
// fetch those from GET /api/usecases/[id]/content instead.
export async function GET(request, { params }) {
  try {
    const { id } = await params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse('Invalid use case ID', 400, 'INVALID_ID', request);
    }

    await dbConnect();

    const useCase = await UseCase.findById(id).lean();
    if (!useCase) {
      return errorResponse('Use case not found', 404, 'NOT_FOUND', request);
    }

    return NextResponse.json({ success: true, data: toUseCaseDTO(useCase) });
  } catch (error) {
    console.error('[GET /api/usecases/[id]] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR', request);
  }
}

// PUT /api/usecases/[id]
// Mongo-backed edit (mint-new for content — see ../_shared for the size cap,
// validation, ref-resolution, and GridFS upload/cleanup helpers shared with
// POST). Partial update: only fields present in the body are touched;
// omitted fields keep their current value.
export async function PUT(request, { params }) {
  // Tracks a newly-uploaded GridFS file (case b: content replaced) so it can
  // be cleaned up if the doc save fails afterward — same mint-then-cleanup
  // shape as POST, plus old-file retirement once the save succeeds (below).
  let newFileId = null;

  try {
    const { id } = await params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse('Invalid use case ID', 400, 'INVALID_ID', request);
    }

    let body;
    try {
      body = await request.json();
    } catch (error) {
      if (error instanceof SyntaxError) {
        return errorResponse('Invalid JSON body', 400, 'INVALID_JSON', request);
      }
      throw error;
    }

    const fieldValidation = validateBasicFields(body);
    if (!fieldValidation.valid) {
      return errorResponse(
        fieldValidation.errors.map((e) => e.message).join('; '),
        400,
        'VALIDATION_ERROR',
        request,
      );
    }

    const { title, description, cover_img, category_id, created_by, tags, content } = body;

    // Distinguish the three content cases via presence of the key itself,
    // not just its value — JSON.stringify drops `undefined` properties, so
    // "key absent after parsing" reliably means the caller never included
    // `content` at all (case a, metadata-only — content_file_id/content_type
    // are left completely untouched). Only an explicitly-present `content`
    // key can mean case (b) (a new notebook string) or case (c) (null/""
    // meaning "clear the notebook"). If a future caller ever sends something
    // that doesn't cleanly fit either — it won't here, since presence is
    // checked directly on the parsed body — this falls back to case (a) by
    // construction rather than guessing.
    const contentProvided = Object.prototype.hasOwnProperty.call(body, 'content');
    const isClearingContent = contentProvided && (content === null || content === '');
    const isReplacingContent = contentProvided && !isClearingContent;

    const anyFieldProvided =
      title !== undefined ||
      description !== undefined ||
      cover_img !== undefined ||
      category_id !== undefined ||
      created_by !== undefined ||
      tags !== undefined ||
      contentProvided;

    if (!anyFieldProvided) {
      return errorResponse('No fields provided to update', 400, 'NO_FIELDS', request);
    }

    // Step 1: validate/size-check new content (case b only) — before
    // touching the DB or GridFS at all.
    let notebookBuffer = null;
    if (isReplacingContent) {
      const validation = validateNotebookContent(content);
      if (!validation.valid) {
        return errorResponse(validation.message, validation.status, validation.code, request);
      }
      notebookBuffer = validation.notebookBuffer;
    }

    await dbConnect();

    const existing = await UseCase.findById(id);
    if (!existing) {
      return errorResponse('Use case not found', 404, 'NOT_FOUND', request);
    }

    // Capture the current file id before anything is mutated — this is
    // what gets retired (or left alone) depending on how the save goes.
    const oldFileId = existing.content_file_id ?? null;

    // Step 2: resolve category/tags/created_by (pure Mongo reads + tag
    // upserts) BEFORE touching GridFS — only for fields actually provided,
    // preserving partial-update semantics. Nothing uploaded yet, so a
    // failure here has nothing to clean up.
    const [categoryRef, createdByRef, tagRefs] = await Promise.all([
      category_id !== undefined ? resolveCategoryRef(category_id) : undefined,
      created_by !== undefined ? resolveCreatedBy(created_by) : undefined,
      tags !== undefined ? resolveTagRefs(tags) : undefined,
    ]);

    // Step 3: mint a new GridFS file for the replacement notebook — the old
    // file (if any) is left completely alone until the doc save below
    // actually succeeds.
    if (isReplacingContent) {
      newFileId = await uploadNotebookToGridFS(notebookBuffer);
    }

    // Step 4: apply only the provided fields, then save.
    if (title !== undefined) existing.title = title.trim();
    if (description !== undefined) existing.description = description;
    if (cover_img !== undefined) existing.cover_img = cover_img;
    if (category_id !== undefined) existing.category = categoryRef;
    // .set() (not direct assignment) for tags — it's a Mongoose
    // DocumentArray, not a plain array, and .set() correctly re-casts a
    // plain array of ref objects into one, same pattern the contributors
    // routes already use for partial updates.
    if (tags !== undefined) existing.set({ tags: tagRefs });
    if (created_by !== undefined) existing.created_by = createdByRef;

    if (isReplacingContent) {
      existing.content_file_id = newFileId;
      existing.content_type = 'notebook';
    } else if (isClearingContent) {
      existing.content_file_id = null;
      existing.content_type = null;
    }
    // else (case a): content_file_id/content_type untouched entirely.

    let saved;
    try {
      saved = await existing.save();
    } catch (saveError) {
      // Step 6: doc save failed — clean up the NEW file (if one was
      // uploaded) and leave the doc + oldFileId completely untouched; the
      // doc still points at a valid old file, so nothing is broken.
      await tryDeleteGridFSFile(newFileId, `orphaned upload after failed update of ${id}`);
      throw saveError;
    }

    // Step 5: save succeeded — now it's safe to retire the OLD file, if
    // this edit replaced or cleared it and it differs from the new one
    // (defensive check; in practice a mint-new replace never reuses the old
    // id, and a clear always sets null, so this is effectively "oldFileId
    // existed and we're in case b or c").
    if (
      (isReplacingContent || isClearingContent) &&
      oldFileId &&
      !(newFileId && oldFileId.equals(newFileId))
    ) {
      await tryDeleteGridFSFile(oldFileId, `retiring replaced content for use case ${id}`);
    }

    return NextResponse.json({ success: true, data: toUseCaseDTO(saved.toObject()) });
  } catch (error) {
    if (error instanceof Error && error.name === 'ValidationError') {
      return errorResponse(error.message, 400, 'VALIDATION_ERROR', request);
    }
    console.error('[PUT /api/usecases/[id]] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR', request);
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
