import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

// Both the failed-attempt counting window and the resulting lockout are 15
// minutes in this design (see src/app/api/library/loginRateLimit.ts), so one
// TTL window does double duty for both see the index comment below.
export const LOGIN_ATTEMPT_WINDOW_SECONDS = 15 * 60;

const loginAttemptSchema = new Schema(
  {
    // The value being tracked: a normalized (lowercased/trimmed) email, or
    // an IP address. Which one is recorded in type.
    key: { type: String, required: true },
    type: { type: String, required: true, enum: ["email", "ip"] },

    attempts: { type: Number, default: 0 },
    first_attempt_at: { type: Date, required: true },
    last_attempt_at: { type: Date, required: true },
  },
  {
    collection: "login_attempts",
  },
);

// One document per tracked key attempts are recorded via an upsert
// keyed on this pair.
loginAttemptSchema.index({ key: 1, type: 1 }, { unique: true });

// TTL index on last_attempt_at (not a fixed expires_at set once at
// first_attempt_at). Mongo's TTL background sweep re-reads the indexed
// field's *current* value on every pass rather than freezing an expiry at
// insert time, so every new failed attempt which bumps last_attempt_at
// pushes the document's expiry forward. That gives SLIDING-window
// semantics for both the 15-minute attempt-counting window and the
// 15-minute lockout: the doc (and with it the attempt count and any active
// lock) only disappears once 15 minutes pass with no further failed
// attempts against that key. A fixed window would instead let attempts
// keep counting indefinitely past 15 minutes, and would let an
// in-progress lockout expire mid-attack sliding is the correct behaviour
// here.
loginAttemptSchema.index(
  { last_attempt_at: 1 },
  { expireAfterSeconds: LOGIN_ATTEMPT_WINDOW_SECONDS },
);

export type LoginAttemptDocument = InferSchemaType<typeof loginAttemptSchema>;

export const LoginAttempt =
  (mongoose.models.LoginAttempt as Model<LoginAttemptDocument>) ||
  mongoose.model<LoginAttemptDocument>("LoginAttempt", loginAttemptSchema);

export default LoginAttempt;
