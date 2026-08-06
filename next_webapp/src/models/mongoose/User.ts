import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

// Embedded, denormalized reference to a role document — same pattern as
// UseCase's CategoryRefSchema: carries the ObjectId ref plus a display copy,
// not just a bare `role_id` scalar.
const RoleRefSchema = new Schema(
  {
    id: { type: Schema.Types.ObjectId, ref: "Role" },
    legacy_id: { type: String, default: null },
    role_name: { type: String, default: null },
  },
  { _id: false },
);

const ProfileSchema = new Schema(
  {
    first_name: { type: String, default: null },
    last_name: { type: String, default: null },
    age: { type: Number, default: null },
    gender: { type: String, default: null },
    profile_img: { type: String, default: null },
    updated_at: { type: Date, default: null },
  },
  { _id: false },
);

const userSchema = new Schema(
  {
    // Old Postgres `user.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    email: { type: String, required: true },
    password: { type: String, required: true },

    role: { type: RoleRefSchema, default: null },
    profile: { type: ProfileSchema, default: null },
  },
  {
    collection: "users",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

userSchema.index({ email: 1 }, { unique: true });
userSchema.index({ "role.id": 1 });
userSchema.index({ legacy_id: 1 });

export type UserDocument = InferSchemaType<typeof userSchema>;

export const User =
  (mongoose.models.User as Model<UserDocument>) ||
  mongoose.model<UserDocument>("User", userSchema);

export default User;
