import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

// No `timestamps` option: the live `logs` collection only has `created_at`,
// not an `updated_at` counterpart (log entries are never updated), so the
// paired createdAt/updatedAt mapping used elsewhere doesn't apply here.
const logSchema = new Schema(
  {
    // Old Postgres `logs.id`, kept as a string during the migration window.
    legacy_id: { type: String, default: null },

    // NOTE: live data has raw ANSI color-escape codes baked into `level` and
    // `message` on 100% of documents (Winston console formatting leaked into
    // persisted records) — see progress report. Not sanitized here; this
    // schema types the field as-is, cleanup is a separate concern.
    level: { type: String, required: true },
    message: { type: String, required: true },
    timestamp: { type: Date, required: true },

    meta: { type: Schema.Types.Mixed, default: null },
    source: { type: String, default: null },

    user_id: { type: Schema.Types.ObjectId, ref: "User", default: null },
    ip_address: { type: String, default: null },
    user_agent: { type: String, default: null },
    method: { type: String, default: null },
    url: { type: String, default: null },
    status_code: { type: Number, default: null },
    // Live data: 0/3477 documents have a non-null response_time — always
    // null in practice today, kept nullable rather than required.
    response_time: { type: Number, default: null },

    created_at: { type: Date, default: Date.now },
  },
  {
    collection: "logs",
  },
);

// Deliberately NOT declaring any index on `timestamp` here. The live
// `logs` collection already has two indexes on that field — a plain
// descending index (`timestamp_-1`) and a TTL index
// (`timestamp_1`, expireAfterSeconds: 7776000) that expires log documents
// after 90 days. Redeclaring either here risks Mongoose treating them as
// out of sync on an autoIndex/syncIndexes pass. Left alone per instructions.
logSchema.index({ user_id: 1 });
logSchema.index({ legacy_id: 1 });

export type LogDocument = InferSchemaType<typeof logSchema>;

export const Log =
  (mongoose.models.Log as Model<LogDocument>) ||
  mongoose.model<LogDocument>("Log", logSchema);

export default Log;
