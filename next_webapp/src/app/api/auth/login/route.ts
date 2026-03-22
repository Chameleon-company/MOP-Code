import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET!;

export async function POST(request: Request) {
    try {
        const { email, password } = await request.json();

        // 1. Validate input
        if (!email || !password) {
            return NextResponse.json(
                { success: false, message: 'Email and password are required' },
                { status: 400 },
            );
        }

        // 2. Find user by email
        const { data: userData, error: userError } = await supabase
            .from('user')
            .select('*')
            .eq('email', email)
            .single();

        if (userError || !userData) {
            return NextResponse.json(
                { success: false, message: 'Invalid email or password' },
                { status: 401 },
            );
        }

        // 3. Compare password with hashed password in DB
        const isPasswordValid = await bcrypt.compare(
            password,
            userData.password,
        );

        if (!isPasswordValid) {
            return NextResponse.json(
                { success: false, message: 'Invalid email or password' },
                { status: 401 },
            );
        }

        // 4. Fetch role details from roles table
        const { data: roleData, error: roleError } = await supabase
            .from('roles')
            .select('*')
            .eq('id', userData.role_id)
            .single();

        if (roleError || !roleData) {
            return NextResponse.json(
                { success: false, message: 'Could not fetch user role' },
                { status: 500 },
            );
        }

        // 5. Fetch user details from user_details table
        const { data: userDetails, error: detailsError } = await supabase
            .from('user_details')
            .select('*')
            .eq('user_id', userData.id)
            .single();

        if (detailsError || !userDetails) {
            return NextResponse.json(
                { success: false, message: 'Could not fetch user details' },
                { status: 500 },
            );
        }

        // 6. Generate JWT token
        const tokenPayload = {
            userId: userData.id,
            email: userData.email,
            roleId: userData.role_id,
            roleName: roleData.role_name,
        };

        const token = jwt.sign(tokenPayload, JWT_SECRET, { expiresIn: '7d' });

        // 7. Return success response with everything
        return NextResponse.json(
            {
                success: true,
                message: 'Login successful',
                data: {
                    userId: userData.id,
                    email: userData.email,
                    firstName: userDetails.first_name,
                    lastName: userDetails.last_name,
                    roleId: userData.role_id,
                    roleName: roleData.role_name,
                    token: token,
                },
            },
            { status: 200 },
        );
    } catch (error) {
        console.error('Login Error:', error);
        return NextResponse.json(
            { success: false, message: 'Internal Server Error' },
            { status: 500 },
        );
    }
}
