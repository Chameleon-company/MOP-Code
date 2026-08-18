import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

const galleryImageSchema = new Schema(
  {
    // Old Postgres `gallery_images.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    title: { type: String, default: null },
    img_url: { type: String, required: true },

    created_by: { type: Schema.Types.ObjectId, ref: "User", default: null },
  },
  {
    collection: "gallery_images",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

galleryImageSchema.index({ legacy_id: 1 });
galleryImageSchema.index({ created_by: 1 });

export type GalleryImageDocument = InferSchemaType<typeof galleryImageSchema>;

export const GalleryImage =
  (mongoose.models.GalleryImage as Model<GalleryImageDocument>) ||
  mongoose.model<GalleryImageDocument>("GalleryImage", galleryImageSchema);

export default GalleryImage;
