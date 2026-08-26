import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import bcrypt from 'bcryptjs';
import { errorResponse } from '@/app/api/library/errorResponse';

export async function POST(request: Request) {
    try {
        const { email, temp_password, new_password, confirm_password } = await request.json();

        // 1. Validate all fields are present
        if (!email || !temp_password || !new_password || !confirm_password) {
            return errorResponse('All fields are required', 400, 'MISSING_FIELDS');
        }

        // 2. Validate new_password matches confirm_password
        if (new_password !== confirm_password) {
            return errorResponse(
                'Passwords do not match',
                400,
                'PASSWORDS_DO_NOT_MATCH',
            );
        }

        // 3. Validate new_password length
        if (new_password.length < 8) {
            return errorResponse(
                'Password must be at least 8 characters',
                400,
                'PASSWORD_TOO_SHORT',
            );
        }

        await dbConnect();

        const normalizeEmail = email.toLowerCase().trim();

        // 4. Look up user in MongoDB
        const userData = await User.findOne({
            email: normalizeEmail,
        }).exec();

        if (!userData) {
            return errorResponse(
                'Invalid credentials',
                401,
                'INVALID_CREDENTIALS',
            );
        }

        // 5. Verify temporary password
        const isTempPasswordValid = await bcrypt.compare(
            temp_password,
            userData.password,
        );

        if (!isTempPasswordValid) {
            return errorResponse(
                'Invalid temporary password',
                401,
                'INVALID_TEMP_PASSWORD',
            );
        }

        // 6. Ensure new password is different from temporary password
        const isSameAsTemp = await bcrypt.compare(
            new_password,
            userData.password,
        );

        if (isSameAsTemp) {
            return errorResponse(
                'New password must be different from temporary password',
                400,
                'SAME_AS_TEMP_PASSWORD',
            );
        }

        // 7. Hash new password
        const hashedPassword = await bcrypt.hash(new_password, 10);

        // 8. Update MongoDB user
        userData.password = hashedPassword;
        await userData.save();

        // 9. Return success
        return NextResponse.json(
            {
                success: true,
                message: 'Password reset successfully',
            },
            { status: 200 },
        );
    } catch (error) {
        console.error('Reset Password Error:', error);
        return errorResponse(
            'Internal Server Error',
            500,
            'INTERNAL_ERROR',
        );
    }
}