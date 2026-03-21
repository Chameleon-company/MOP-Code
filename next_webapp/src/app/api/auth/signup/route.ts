import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';

export async function POST(request: Request) {
    try {
        const { firstName, lastName, email, password } = await request.json();

        // Validate input
        if (!firstName || !lastName || !email || !password) {
            return NextResponse.json(
                { success: false, message: 'All fields are required' },
                { status: 400 }
            );
        }

        // Check if user already exists
        const { data: existingUser, error: fetchError } = await supabase
            .from('user')
            .select('*')
            .eq('email', email)
            .single();

        if (existingUser) {
            return NextResponse.json(
                { success: false, message: 'User already exists' },
                { status: 400 }
            );
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Insert into user table
        const { data: userData, error: userError } = await supabase
            .from('user')
            .insert([
                {
                    email: email,
                    password: hashedPassword,
                    role_id: 1 
                }
            ])
            .select()
            .single();

        if (userError) {
            throw userError;
        }

        // Insert into user_details table
        const { error: detailsError } = await supabase
            .from('user_details')
            .insert([
                {
                    user_id: userData.id,
                    first_name: firstName,
                    last_name: lastName
                }
            ]);

        if (detailsError) {
            throw detailsError;
        }

        return NextResponse.json(
            { success: true, message: 'User registered successfully' },
            { status: 201 }
        );

    } catch (error) {
        console.error('Signup Error:', error);

        return NextResponse.json(
            { success: false, message: 'Internal Server Error bla' },
            { status: 500 }
        );
    }
}