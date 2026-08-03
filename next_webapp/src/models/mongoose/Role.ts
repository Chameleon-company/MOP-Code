import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

const roleSchema = new Schema(
  {
    // Old Postgres `roles.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },
    role_name: { type: String, required: true },
  },
  {
    collection: "roles",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

roleSchema.index({ legacy_id: 1 });
roleSchema.index({ role_name: 1 }, { unique: true });

export type RoleDocument = InferSchemaType<typeof roleSchema>;

export const Role =
  (mongoose.models.Role as Model<RoleDocument>) ||
  mongoose.model<RoleDocument>("Role", roleSchema);

export default Role;
