import { NextResponse } from 'next/server';
import dbConnect from '../../../../lib/mongodb'; // Import dbConnect to initialize the database connection
import User from '../../../../models/User'; // Import the Mongoose User model

// Define the handler for POST requests
export async function POST(request: Request) {
    console.log('Received POST request at /api/login');

    try {
        // Parse the request body
        const { email, password } = await request.json();
        console.log('Login request body:', { email });

        // Validate input
        if (!email || !password) {
            return NextResponse.json({ success: false, message: 'Email and password are required.' }, { status: 400 });
        }

        // Connect to MongoDB
        await dbConnect(); // This initializes the connection but doesn't return anything directly
        console.log('Connected to MongoDB');

        const user = await User.findOne({ email });
        if (!user) {
            return NextResponse.json({ success: false, message: 'Invalid email or password.' }, { status: 401 });
        }
        console.log('User authenticated successfully:', user.email);

        return NextResponse.json({
            success: true,
            message: 'Login successful.',
            user: {
                id: user._id,
                name: `${user.firstName} ${user.lastName}`,
                email: user.email
            }
        }, { status: 200 });

    } catch (error) {
        console.error('Login error:', error);
        return NextResponse.json({ success: false, message: 'Internal Server Error' }, { status: 500 });
    }
}