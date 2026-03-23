import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { errorResponse } from '@/app/api/library/errorResponse';

const JWT_SECRET = process.env.JWT_SECRET!;

export async function POST(request: Request) {
    try {
        const { email, password } = await request.json();

        // 1. Validate input
        if (!email || !password) {
            return errorResponse('Email and password are required', 400, 'MISSING_FIELDS');
        }

        // 2. Find user by email
        const { data: userData, error: userError } = await supabase
            .from('user')
            .select('*')
            .eq('email', email)
            .single();

        if (userError || !userData) {
            return errorResponse('Invalid email or password', 401, 'INVALID_CREDENTIALS');
        }

        // 3. Compare password with hashed password in DB
        const isPasswordValid = await bcrypt.compare(
            password,
            userData.password,
        );

        if (!isPasswordValid) {
            return errorResponse('Invalid email or password', 401, 'INVALID_CREDENTIALS');
        }

        // 4. Fetch role details from roles table
        const { data: roleData, error: roleError } = await supabase
            .from('roles')
            .select('*')
            .eq('id', userData.role_id)
            .single();

        if (roleError || !roleData) {
            return errorResponse('Could not fetch user role', 500, 'ROLE_FETCH_ERROR');
        }

        // 5. Fetch user details from user_details table
        const { data: userDetails, error: detailsError } = await supabase
            .from('user_details')
            .select('*')
            .eq('user_id', userData.id)
            .single();

        if (detailsError || !userDetails) {
            return errorResponse('Could not fetch user details', 500, 'DETAILS_FETCH_ERROR');
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
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
