import clientPromise from '../../../../lib/mongodb';
export async function GET(request) {
  try {
    const client = await clientPromise;
    
    // Ping MongoDB to check the connection
    await client.db('admin').command({ ping: 1 });
    //console.log("Connected to MongoDB");

    // Fetch use cases
    const db = client.db('mop');  // database name
    const collection = db.collection('mop-usecases');  // collection name
    const usecases = await collection.find({}).toArray();

    // Ensuring the response is an array
    const response = Array.isArray(usecases) ? usecases : [usecases];

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('MongoDB Connection Error:', error);
    
    // Returning an error response with status 500
    return new Response(JSON.stringify({ message: 'Failed to fetch use cases' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
