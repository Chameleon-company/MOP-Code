// import dbConnect from '../../../../lib/mongodb';
// import clientPromise from '../../../../lib/mongodb';
import clientPromise from '../../../../lib/mongodb';
import UseCase from '../../../../models/UseCase';
import { NextResponse } from 'next/server';

import { MongoClient } from 'mongodb';




// export async function GET(request) {
//   try {
//     const client = await clientPromise;
    
//     // Ping MongoDB to check the connection
//     await client.db('admin').command({ ping: 1 });
//     //console.log("Connected to MongoDB");

//     // Fetch use cases
//     const db = client.db('mop');  // database name
//     const collection = db.collection('mop-usecases');  // collection name
//     const usecases = await collection.find({}).toArray();

//     // Ensuring the response is an array
//     //const response = Array.isArray(usecases) ? usecases : [usecases];

//     //Mapping over the usecases
//     const response = usecases.map(usecase => ({
//         _id: Number(usecase._id),
//         name: String(usecase.name),
//         description: String(usecase.description),
//         tags: Array.isArray(usecase.tags) ? usecase.tags.map(String) : [], 
//         filename: usecase.filename || "placeholder name"
//       }));

//     return new Response(JSON.stringify(response), {
//       status: 200,
//       headers: { 'Content-Type': 'application/json' },
//     });
//   } catch (error) {
//     console.error('MongoDB Connection Error:', error);
    
//     // Returning an error response with status 500
//     return new Response(JSON.stringify({ message: 'Failed to fetch use cases' }), {
//       status: 500,
//       headers: { 'Content-Type': 'application/json' },
//     });
//   }
// }

// POST method to uplad use cases to mongoDB
export async function POST(request: Request) {

  console.log('meow meow');

  
  // Ping MongoDB to check the connection
  // await client.db('admin').command({ ping: 1 });
  const { title, description, tags, filename } = await request.json();
  
  try {
    // const client = await clientPromise;
    const url = process.env.MONGODB_URI;
    const dbName = 'useCasesDB';

    MongoClient.connect(url, function(err, client) {
      console.log("Connected successfully to server");
      const db = client.db(dbName);

    // const db = client.db(dbName);
    // const collection = db.collection('useCasesCollection');
    // dbConnect();
    
    const newUseCase = new UseCase({ 
      name: title, 
      description: description, 
      tags: tags, 
      filename: filename }
    );

    // const db = client.db('useCasesDB');
    // await newUseCase.save();
    // await collection.insertOne(newUseCase);
    
    db.collection('useCasesCollection').insertOne(newUseCase); 

    client.close();
  });
    return NextResponse.json({ message: "Usecase created" }, { status:201 });


    // await newUseCase.save();
    // await UseCase.create({ name, description, tags, filename });
    // return response.insertedId;

  }
  catch (error) {
    console.log(error);
  }
}
