import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import bcrypt from 'bcryptjs';
import { errorResponse } from '@/app/api/library/errorResponse';
import { checkPasswordResetRateLimit, recordPasswordResetAttempt, clearPasswordResetAttempts, getClientIp } from '@/app/api/library/passwordResetRateLimit';

export async function POST(request: Request) {
    try {
        const ip = getClientIp(request);
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

        const normalizeEmail = email.toLowerCase().trim();

        const { limited } = await checkPasswordResetRateLimit(normalizeEmail, ip, "failed_reset_attempt");
        if (limited) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse('Too many failed reset attempts, please try again later', 429, 'TOO_MANY_ATTEMPTS');
        }
        // 4. Look up user in MongoDB
        const userData = await User.findOne({
            email: normalizeEmail,
        }).exec();

        if (!userData) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse(
                'Invalid credentials',
                401,
                'INVALID_CREDENTIALS',
            );
        }

        // 5. Verify temporary password token
        if (!userData.reset_token) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse(
                'No reset token found',
                401,
                'INVALID_TEMP_PASSWORD',
            );
        }

        if (userData.reset_token_used) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse(
                'Temporary password has already been used',
                401,
                'TOKEN_USED',
            );
        }

        if (userData.reset_token_expires && new Date() > userData.reset_token_expires) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse(
                'Temporary password has expired',
                401,
                'TOKEN_EXPIRED',
            );
        }

        const isTempPasswordValid = await bcrypt.compare(
            temp_password,
            userData.reset_token,
        );

        if (!isTempPasswordValid) {
            await recordPasswordResetAttempt(normalizeEmail, ip, "failed_reset_attempt");
            return errorResponse(
                'Invalid temporary password',
                401,
                'INVALID_TEMP_PASSWORD',
            );
        }

        // 6. Ensure new password is different from temporary password
        const isSameAsTemp = await bcrypt.compare(
            new_password,
            userData.reset_token,
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
        userData.reset_token_used = true;
        await userData.save();

        await clearPasswordResetAttempts(normalizeEmail, ip, "failed_reset_attempt");

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