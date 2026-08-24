import mongoose from "mongoose";
import Category from "@/models/mongoose/Category";
import Tag from "@/models/mongoose/Tag";
import User from "@/models/mongoose/User";
import { getGridFSBucket } from "@/lib/gridfs";

// Shared between POST /api/usecases (create) and PUT /api/usecases/[id]
// (edit) — kept here so the two routes can never silently diverge on
// validation, ref-resolution, or GridFS mint/cleanup semantics.

// Server-side cap on notebook JSON size, enforced BEFORE any GridFS write —
// measured on the actual byte length (Buffer.byteLength), not .length,
// since .length counts UTF-16 code units and would under-count for any
// multi-byte characters in the notebook (cell text, unicode in outputs, etc).
export const MAX_NOTEBOOK_BYTES = 60 * 1024 * 1024; // 60 MB

// Resolve an embedded category ref from an incoming category_id. The admin
// add/edit forms still source category_id from /api/categories, which is
// still Supabase-backed (a Postgres numeric id) — there is no live Mongo
// Category collection being written to yet. Try a real Mongo ObjectId first
// (forward-compatible with a future Mongo-backed category picker), then
// fall back to Category.legacy_id (the string field this migration already
// carries the same numeric id under — see Category.ts). If neither
// resolves, still keep the id as a legacy-only reference rather than
// dropping it — same "carry the id even without a live match" pattern used
// elsewhere in this migration (UseCase.legacy_id, TagRefSchema).
export async function resolveCategoryRef(categoryId) {
  if (categoryId === undefined || categoryId === null || categoryId === "") {
    return null;
  }
  const idStr = String(categoryId);

  let category = null;
  if (mongoose.Types.ObjectId.isValid(idStr)) {
    category = await Category.findById(idStr).lean();
  }
  if (!category) {
    category = await Category.findOne({ legacy_id: idStr }).lean();
  }

  if (category) {
    return {
      id: category._id,
      legacy_id: category.legacy_id ?? idStr,
      category_name: category.category_name ?? null,
    };
  }

  return { id: null, legacy_id: idStr, category_name: null };
}

// Resolve created_by (a legacy Postgres numeric user id, read out of the
// client's localStorage session) to a Mongo User ObjectId, same
// legacy-id-fallback approach as resolveCategoryRef. Unlike category/tags,
// this field has no legacy_id sibling on the UseCase schema itself (it's a
// bare `Schema.Types.ObjectId, ref: "User"`), so an id that resolves to
// nothing is simply dropped (stored as null) rather than blocking the
// request — there's nowhere on this field to keep the raw legacy id if it
// doesn't resolve.
export async function resolveCreatedBy(createdBy) {
  if (createdBy === undefined || createdBy === null || createdBy === "") {
    return null;
  }
  const idStr = String(createdBy);

  if (mongoose.Types.ObjectId.isValid(idStr)) {
    const user = await User.findById(idStr).select("_id").lean();
    if (user) return user._id;
  }

  const user = await User.findOne({ legacy_id: idStr }).select("_id").lean();
  return user ? user._id : null;
}

// Find-or-create a Tag document for each incoming tag name, then embed a
// denormalized ref on the doc — the Mongo equivalent of the old
// find-or-create-then-link-via-usecase_tags logic, minus the join table
// (tags live directly on the doc now, see TagRefSchema in UseCase.ts).
export async function resolveTagRefs(tagNames) {
  if (!Array.isArray(tagNames) || tagNames.length === 0) return [];

  const refs = [];
  const seenSlugs = new Set();

  for (const raw of tagNames) {
    if (typeof raw !== "string" || raw.trim().length === 0) continue;

    const name = raw.trim();
    const slug = name.toLowerCase().replace(/\s+/g, "-");
    if (seenSlugs.has(slug)) continue;
    seenSlugs.add(slug);

    // Atomic find-or-create — replaces the old insert-then-catch-23505-
    // then-refetch dance, which Mongo's upsert makes unnecessary.
    const tag = await Tag.findOneAndUpdate(
      { slug },
      { $setOnInsert: { name, slug } },
      { upsert: true, new: true },
    ).lean();

    refs.push({
      id: tag._id,
      legacy_id: tag.legacy_id ?? null,
      name: tag.name,
      slug: tag.slug,
    });
  }

  return refs;
}

// Validate a notebook `content` string: must be a string, within the size
// cap, valid JSON, and shaped like a notebook (an object with a `cells`
// array) — exactly the checks POST has always applied, factored out so PUT
// applies the identical checks for a content-replacing edit. Returns a
// plain result object instead of calling errorResponse itself, so each
// route can build its own error response the way it already does (message/
// status/code are identical either way).
export function validateNotebookContent(content) {
  if (typeof content !== "string") {
    return {
      valid: false,
      message: "content must be a JSON string",
      status: 400,
      code: "INVALID_CONTENT",
    };
  }

  const contentBytes = Buffer.byteLength(content, "utf8");
  if (contentBytes > MAX_NOTEBOOK_BYTES) {
    return {
      valid: false,
      message: `Notebook content exceeds the ${MAX_NOTEBOOK_BYTES / (1024 * 1024)} MB limit`,
      status: 413,
      code: "CONTENT_TOO_LARGE",
    };
  }

  let parsedNotebook;
  try {
    parsedNotebook = JSON.parse(content);
  } catch (error) {
    if (error instanceof SyntaxError) {
      return {
        valid: false,
        message: "content is not valid notebook JSON",
        status: 400,
        code: "INVALID_NOTEBOOK_JSON",
      };
    }
    throw error;
  }

  if (
    !parsedNotebook ||
    typeof parsedNotebook !== "object" ||
    !Array.isArray(parsedNotebook.cells)
  ) {
    return {
      valid: false,
      message: "content must be a notebook JSON object with a cells array",
      status: 400,
      code: "INVALID_NOTEBOOK_FORMAT",
    };
  }

  return {
    valid: true,
    contentType: "notebook",
    notebookBuffer: Buffer.from(content, "utf8"),
  };
}

// Upload a validated notebook buffer to GridFS, resolving once the write is
// fully persisted (files + chunks docs both written), with the new file's
// id. The id itself is available synchronously on the stream (the driver
// assigns it in the stream's constructor, before any bytes are written),
// but callers should treat the awaited resolution as "safe to reference
// this id from a saved doc."
export async function uploadNotebookToGridFS(notebookBuffer) {
  const bucket = await getGridFSBucket();
  const uploadStream = bucket.openUploadStream(`usecase-${Date.now()}.ipynb`, {
    metadata: { content_type: "notebook" },
  });
  const fileId = uploadStream.id;

  await new Promise((resolve, reject) => {
    uploadStream.on("finish", resolve);
    uploadStream.on("error", reject);
    uploadStream.end(notebookBuffer);
  });

  return fileId;
}

// Best-effort GridFS delete, used both for "clean up a newly orphaned
// upload after a failed doc save" (create and edit) and "retire an old file
// after a successful replace/clear" (edit only). Never throws — a cleanup
// failure is logged (leaving a harmless orphaned file) rather than allowed
// to fail or mask the outcome of the request that triggered it.
export async function tryDeleteGridFSFile(fileId, context) {
  if (!fileId) return;
  try {
    const bucket = await getGridFSBucket();
    await bucket.delete(fileId);
  } catch (cleanupError) {
    console.error(
      `[usecases] failed to delete GridFS file ${fileId} (${context}):`,
      cleanupError,
    );
  }
}
