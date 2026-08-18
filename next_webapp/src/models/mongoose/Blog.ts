import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

const blogSchema = new Schema(
  {
    // Old Postgres `blogs.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    title: { type: String, required: true },
    // No size cap: largest live document is ~3.5KB (see progress report),
    // nowhere near the 16MB BSON document limit, but nothing here enforces
    // an upper bound if that changes.
    content: { type: String, default: null },
    description: { type: String, default: null },
    cover_img: { type: String, default: null },
    published_date: { type: Date, default: null },

    created_by: { type: Schema.Types.ObjectId, ref: "User", default: null },
  },
  {
    collection: "blogs",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

blogSchema.index({ legacy_id: 1 });
blogSchema.index({ created_by: 1 });
// Mirrors the existing live index on `published_date` (descending).
blogSchema.index({ published_date: -1 });

export type BlogDocument = InferSchemaType<typeof blogSchema>;

export const Blog =
  (mongoose.models.Blog as Model<BlogDocument>) ||
  mongoose.model<BlogDocument>("Blog", blogSchema);

export default Blog;
