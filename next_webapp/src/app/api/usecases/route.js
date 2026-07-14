// Telling Next.js this file needs the Node.js runtime 
export const runtime = 'nodejs';

import { MongoClient } from 'mongodb';

// URI from env
const uri = process.env.MONGODB_URI;
if (!uri) {
  throw new Error('Please define the MONGODB_URI environment variable');
}

// Cache the client across hot-reloads in development
let cachedClient = global._mongoClient;
if (!cachedClient) {
  cachedClient = new MongoClient(uri);
  global._mongoClient = cachedClient;
}

export async function GET(request) {
  try {
    // Ensuring the client is connected
    if (!cachedClient.isConnected && typeof cachedClient.connect === 'function') {
      await cachedClient.connect();
    }

    // Switch to your database and collection
    const db = cachedClient.db('mop');
    const coll = db.collection('mop-usecases');
    const docs = await coll.find({}).toArray();

    // Shape the response
    const response = docs.map((uc) => ({
      id: Number(uc._id),
      name: String(uc.name),
      description: String(uc.description),
      tags: Array.isArray(uc.tags) ? uc.tags.map(String) : [],
      filename: String(uc.filename || ""),
    }));

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("MongoDB Connection Error:", error);
    return new Response(
      JSON.stringify({ message: "Failed to fetch use cases" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}


