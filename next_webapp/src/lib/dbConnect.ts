import mongoose, { type Connection } from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI;

interface MongooseCache {
  conn: Connection | null;
  promise: Promise<Connection> | null;
}

declare global {
  // eslint-disable-next-line no-var
  var __mongooseCache: MongooseCache | undefined;
}

const cached: MongooseCache = global.__mongooseCache ?? {
  conn: null,
  promise: null,
};

if (!global.__mongooseCache) {
  global.__mongooseCache = cached;
}

/**
 * Cached Mongoose connection helper. Reuses a single connection pool across
 * hot reloads (dev) and warm serverless invocations instead of opening a new
 * one per call/import.
 */
export async function dbConnect(): Promise<Connection> {
  if (!MONGODB_URI) {
    throw new Error(
      "Please define the MONGODB_URI environment variable inside .env",
    );
  }

  if (cached.conn) {
    return cached.conn;
  }

  if (!cached.promise) {
    cached.promise = mongoose
      .connect(MONGODB_URI, {
        bufferCommands: false,
        serverSelectionTimeoutMS: 10000,
      })
      .then((mongooseInstance) => mongooseInstance.connection);
  }

  try {
    cached.conn = await cached.promise;
  } catch (error) {
    // Allow the next call to retry instead of replaying a rejected promise.
    cached.promise = null;
    throw error;
  }

  return cached.conn;
}

export default dbConnect;
