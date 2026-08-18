import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

const categorySchema = new Schema(
  {
    // Old Postgres `categories.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    category_name: { type: String, required: true },
    description: { type: String, default: null },
    cover_img: { type: String, default: null },

    created_by: { type: Schema.Types.ObjectId, ref: "User", default: null },
  },
  {
    collection: "categories",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

categorySchema.index({ legacy_id: 1 });
categorySchema.index({ created_by: 1 });

export type CategoryDocument = InferSchemaType<typeof categorySchema>;

export const Category =
  (mongoose.models.Category as Model<CategoryDocument>) ||
  mongoose.model<CategoryDocument>("Category", categorySchema);

export default Category;
