import clientPromise from '../../../../lib/mongodb';
import mongoose from 'mongoose';

// MongoDB connection logic
let isConnected = false;

export const config = {
  api: {
    bodyParser: true, // Allow the Next.js body parser
  },
};

// Mongoose model for the UseCase
const useCaseSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: { type: String, required: true },
  trimester: { type: String, required: true },
  tags: { type: [String], required: true },
  filename: { type: String, required: true },
});

const UseCase = mongoose.models.UseCase || mongoose.model('UseCase', useCaseSchema);

export async function POST(request) {
  try {
    // Connect to MongoDB if not already connected
    if (mongoose.connection.readyState !== 1) {
      await mongoose.connect("mongodb://localhost:27017/admin", {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      });
    }

    // Parse incoming JSON body
    const formData = await request.json();

    const { name, description, trimester, tags, filePath } = formData;

    // Ensure all required fields are provided
    if (!name || !description || !trimester || !tags || !filePath) {
      return new Response(
        JSON.stringify({ message: 'All fields are required.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Generate unique filename if necessary
    const filename = filePath;

    // Prepare use case document
    const newUsecase = {
      name: String(name),
      description: String(description),
      trimester: String(trimester),
      tags: Array.isArray(tags) ? tags.map(String) : [],
      filename: filename,
    };

    // Insert into MongoDB collection
    const result = await UseCase.create(newUsecase);
    console.log("result",result)
    // Return success response
    return new Response(
      JSON.stringify({
        message: 'Use case added successfully!',
        id: result._id,
      }),
      {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Upload error:', error);
    return new Response(
      JSON.stringify({
        message: 'Failed to add use case',
        error: error.message,
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

