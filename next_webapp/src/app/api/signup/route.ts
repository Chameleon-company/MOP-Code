// app/api/signup/route.ts

import { NextResponse } from 'next/server';
import dbConnect from '../../../../lib/postgresql'; // Import dbConnect to initialize the database connection
import User from '../../../../models/User'; // Import the Mongoose User model

// Define the handler for POST requests
export async function POST(request: Request) {
    console.log('Received POST request at /api/signup');

    try {
        // Parse the request body
        const { firstName, lastName, email, password } = await request.json();
        console.log('Request body:', { firstName, lastName, email, password });

        // Validate input
        if (!firstName || !lastName || !email || !password) {
            return NextResponse.json({ success: false, message: 'All fields are required.' }, { status: 400 });
        }

        // Connect to MongoDB
        await dbConnect(); // This initializes the connection but doesn't return anything directly
        console.log('Connected to MongoDB');

        // Check if the user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return NextResponse.json({ success: false, message: 'User already exists.' }, { status: 400 });
        }

        // Create a new user
        const newUser = new User({
            firstName,
            lastName,
            email,
            password, // Ideally, hash the password before saving
        });

        // Save the new user to the database
        await newUser.save();
        console.log('User created successfully:', newUser);

        return NextResponse.json({ success: true, message: 'User created successfully.' }, { status: 201 });
    } catch (error) {
        console.error('Error creating user:', error);
        return NextResponse.json({ success: false, message: 'Internal Server Error' }, { status: 500 });
    }
}

