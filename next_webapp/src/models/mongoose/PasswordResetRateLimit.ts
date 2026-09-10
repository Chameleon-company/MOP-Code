import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

export const PASSWORD_RESET_WINDOW_SECONDS = 15 * 60;

const passwordResetRateLimitSchema = new Schema(
  {
    key: { type: String, required: true },
    type: { type: String, required: true, enum: ["email", "ip"] },
    action: { type: String, required: true, enum: ["forgot_password_request", "failed_reset_attempt"] },

    attempts: { type: Number, default: 0 },
    first_attempt_at: { type: Date, required: true },
    last_attempt_at: { type: Date, required: true },
  },
  {
    collection: "password_reset_rate_limits",
  },
);

passwordResetRateLimitSchema.index({ key: 1, type: 1, action: 1 }, { unique: true });

passwordResetRateLimitSchema.index(
  { last_attempt_at: 1 },
  { expireAfterSeconds: PASSWORD_RESET_WINDOW_SECONDS },
);

export type PasswordResetRateLimitDocument = InferSchemaType<typeof passwordResetRateLimitSchema>;

export const PasswordResetRateLimit =
  (mongoose.models.PasswordResetRateLimit as Model<PasswordResetRateLimitDocument>) ||
  mongoose.model<PasswordResetRateLimitDocument>("PasswordResetRateLimit", passwordResetRateLimitSchema);

export default PasswordResetRateLimit;
