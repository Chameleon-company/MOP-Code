import { NextResponse } from 'next/server';
import supabase from '../../../../lib/postgresql';

// Define the request type for clarity
interface LoginRequest {
  email: string;
  password: string;
}

export async function POST(request: Request) {
  try {
    const body: LoginRequest = await request.json();

    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json(
        { success: false, message: 'Email and password are required.' },
        { status: 400 }
      );
    }

    const { data: users, error } = await supabase
      .from('users')
      .select('*')
      .eq('email', email)
      .limit(1);

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json(
        { success: false, message: 'Database error.' },
        { status: 500 }
      );
    }

    const user = users?.[0] ?? null;

    // Plain text password check (replace with hash check in production)
    if (!user || user.password !== password) {
      return NextResponse.json(
        { success: false, message: 'Invalid email or password.' },
        { status: 401 }
      );
    }

    return NextResponse.json(
      {
        success: true,
        message: 'Login successful.',
        user: {
          id: user.id || user._id,
          name: `${user.first_name ?? ''} ${user.last_name ?? ''}`.trim(),
          email: user.email,
        },
      },
      { status: 200 }
    );
  } catch (err) {
    console.error('Login error:', err);
    return NextResponse.json(
      { success: false, message: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
