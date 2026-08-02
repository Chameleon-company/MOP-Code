import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

// No `timestamps` option: the live `tags` collection has no created_at/updated_at
// fields at all (confirmed against all 128 documents) — do not invent them.
const tagSchema = new Schema(
  {
    // Old Postgres `tags.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    name: { type: String, required: true },
    slug: { type: String, required: true },
  },
  {
    collection: "tags",
  },
);

tagSchema.index({ legacy_id: 1 });
// Mirrors the existing live unique index on `slug` — confirmed 0 duplicate
// slugs across all 128 live documents before adding this.
tagSchema.index({ slug: 1 }, { unique: true });

export type TagDocument = InferSchemaType<typeof tagSchema>;

export const Tag =
  (mongoose.models.Tag as Model<TagDocument>) ||
  mongoose.model<TagDocument>("Tag", tagSchema);

export default Tag;
