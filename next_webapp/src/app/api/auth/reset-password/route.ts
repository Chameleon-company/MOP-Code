import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
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
            return errorResponse('Passwords do not match', 400, 'PASSWORDS_DO_NOT_MATCH');
        }

        // 3. Validate new_password length
        if (new_password.length < 8) {
            return errorResponse('Password must be at least 8 characters', 400, 'PASSWORD_TOO_SHORT');
        }

        // 4. Look up user by email
        const { data: userData, error: userError } = await supabase
            .from('user')
            .select('id, email, password')
            .eq('email', email)
            .single();

        if (userError || !userData) {
            return errorResponse('Invalid credentials', 401, 'INVALID_CREDENTIALS');
        }

        // 5. Verify temp_password against stored hash
        const isTempPasswordValid = await bcrypt.compare(temp_password, userData.password);

        if (!isTempPasswordValid) {
            return errorResponse('Invalid temporary password', 401, 'INVALID_TEMP_PASSWORD');
        }

        // 6. Ensure new_password is not identical to temp_password
        const isSameAsTemp = await bcrypt.compare(new_password, userData.password);

        if (isSameAsTemp) {
            return errorResponse(
                'New password must be different from temporary password',
                400,
                'SAME_AS_TEMP_PASSWORD',
            );
        }

        // 7. Hash new password
        const hashedPassword = await bcrypt.hash(new_password, 10);

        // 8. Update user record with new hashed password
        const { error: updateError } = await supabase
            .from('user')
            .update({
                password: hashedPassword,
                updated_at: new Date().toISOString(),
            })
            .eq('email', email);

        if (updateError) {
            throw updateError;
        }

        // 9. Return success
        return NextResponse.json(
            { success: true, message: 'Password reset successfully' },
            { status: 200 },
        );
    } catch (error) {
        console.error('Reset Password Error:', error);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
