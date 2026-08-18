import mongoose from "mongoose";
import dbConnect from "./dbConnect";

/**
 * Bucket name for notebook file storage. Referenced from here rather than
 * as a bare string so every route/helper that touches GridFS agrees on it.
 */
export const NOTEBOOKS_BUCKET_NAME = "notebooks";

/**
 * Returns a GridFSBucket for notebook storage, calling dbConnect()
 * internally so callers do one await and get a ready bucket.
 */
export async function getGridFSBucket(): Promise<mongoose.mongo.GridFSBucket> {
  const conn = await dbConnect();

  if (!conn.db) {
    throw new Error("MongoDB connection has no db handle (not yet connected).");
  }

  return new mongoose.mongo.GridFSBucket(conn.db, {
    bucketName: NOTEBOOKS_BUCKET_NAME,
  });
}

export default getGridFSBucket;
