import { NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { errorResponse } from '@/app/api/library/errorResponse';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';

const JWT_SECRET = process.env.JWT_SECRET!;

export async function POST(request: Request) {
    try {
        // what changed on sprint 2 T2 2026
        await dbConnect();

        const { email, password } = await request.json();

        // 1. Validate input
        if (!email || !password) {
            return errorResponse('Email and password are required', 400, 'MISSING_FIELDS');
        }
        const normalizeEmail = email.toLowerCase().trim();
        // 2. Find user by email
        const userData = await User.findOne({ email:normalizeEmail }).exec();

        if (!userData) {
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

        // 4. Fetch role details from embedded role object
        const roleData = userData.role;

        if (!roleData) {
            return errorResponse('Could not fetch user role', 500, 'ROLE_FETCH_ERROR');
        }

        // 5. Fetch user details from embedded profile object
        const userDetails = userData.profile;

        if (!userDetails) {
            return errorResponse('Could not fetch user details', 500, 'DETAILS_FETCH_ERROR');
        }

        const userId = String(userData._id);

        const roleId = roleData.legacy_id
            ? Number(roleData.legacy_id)
            : null;

        // 6. Generate JWT token
        const tokenPayload = {
            userId: userId,
            email: userData.email,
            roleId: roleId,
            roleName: roleData.role_name,
        };

        const token = jwt.sign(tokenPayload, JWT_SECRET, { expiresIn: '7d' });

        // 7. Return success response with everything
        return NextResponse.json(
            {
                success: true,
                message: 'Login successful',
                data: {
                    userId: userId,
                    email: userData.email,
                    firstName: userDetails.first_name,
                    lastName: userDetails.last_name,
                    roleId: roleId,
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