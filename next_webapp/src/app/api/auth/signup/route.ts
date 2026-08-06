import { NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import { errorResponse } from '@/app/api/library/errorResponse';
import logger from '@/utils/logger';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import Role from '@/models/mongoose/Role';

export async function POST(request: Request) {
    try {
        await dbConnect();
        
        const { firstName, lastName, email, password } = await request.json();
        // Validate input
        if (!firstName || !lastName || !email || !password) {
            return errorResponse('All fields are required', 400, 'MISSING_FIELDS');
        }
        const normalizeEmail = email.toLowerCase().trim();
        // Check if user already exists
        const existingUser = await User.findOne({ email:normalizeEmail }).exec();

        if (existingUser) {
            return errorResponse('User already exists', 400, 'USER_EXISTS');
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);


        // Fetch the default regular user role
        const roleData = await Role.findOne({ legacy_id: '2' }).exec();

        if (!roleData) {
            throw new Error('Default user role could not be found');
        }

        // Insert user with embedded role and profile
        await User.create({
            email: normalizeEmail,
            password: hashedPassword,
            role: {
                id: roleData._id,
                legacy_id: roleData.legacy_id,
                role_name: roleData.role_name,
            },
            profile: {
                first_name: firstName,
                last_name: lastName,
            },
        });

        return NextResponse.json(
            { success: true, message: 'User registered successfully' },
            { status: 201 }
        );

    } catch (error) {
        logger.error(`Signup Error: ${error instanceof Error ? error.message : String(error)}`);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR', request);
    }
}