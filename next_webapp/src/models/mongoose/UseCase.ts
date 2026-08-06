import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

// Embedded, denormalized reference to a category document — not populated
// via a Mongoose `ref` walk by default, just carries the id + a display copy.
const CategoryRefSchema = new Schema(
  {
    id: { type: Schema.Types.ObjectId, ref: "Category" },
    legacy_id: { type: String, default: null },
    category_name: { type: String, default: null },
  },
  { _id: false },
);

// Embedded, denormalized reference to a tag document.
const TagRefSchema = new Schema(
  {
    id: { type: Schema.Types.ObjectId, ref: "Tag" },
    legacy_id: { type: String, default: null },
    name: { type: String, default: null },
    slug: { type: String, default: null },
  },
  { _id: false },
);

const useCaseSchema = new Schema(
  {
    // Old Postgres `usecases.id`, kept as a string during the migration
    // window so legacy references (e.g. `?legacy_id=24`) still resolve.
    legacy_id: { type: String, default: null },

    title: { type: String, required: true },
    description: { type: String, default: null },
    cover_img: { type: String, default: null },

    // Reference to a GridFS file holding the notebook/HTML body. Deliberately
    // NOT a Mongoose `ref` — GridFS files aren't a model, they're chunks in
    // `<bucket>.files`/`<bucket>.chunks`, looked up manually via GridFSBucket.
    content_file_id: { type: Schema.Types.ObjectId, default: null },
    content_type: {
      type: String,
      enum: ["notebook", "html", null],
      default: null,
    },

    category: { type: CategoryRefSchema, default: null },
    tags: { type: [TagRefSchema], default: [] },

    created_by: { type: Schema.Types.ObjectId, ref: "User", default: null },
  },
  {
    collection: "usecases",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

useCaseSchema.index({ "category.id": 1 });
useCaseSchema.index({ "tags.slug": 1 });
useCaseSchema.index({ created_by: 1 });
useCaseSchema.index({ legacy_id: 1 });

export type UseCaseDocument = InferSchemaType<typeof useCaseSchema>;

export const UseCase =
  (mongoose.models.UseCase as Model<UseCaseDocument>) ||
  mongoose.model<UseCaseDocument>("UseCase", useCaseSchema);

export default UseCase;
