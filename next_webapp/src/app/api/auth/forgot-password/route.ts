import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';
import nodemailer from 'nodemailer';
import { errorResponse } from '@/app/api/library/errorResponse';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const TEMP_PASSWORD_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
const TEMP_PASSWORD_LENGTH = 10;

function generateTempPassword(): string {
    let result = '';
    for (let i = 0; i < TEMP_PASSWORD_LENGTH; i++) {
        result += TEMP_PASSWORD_CHARS.charAt(
            Math.floor(Math.random() * TEMP_PASSWORD_CHARS.length),
        );
    }
    return result;
}

const SAFE_RESPONSE = NextResponse.json(
    { success: true, message: 'If this email exists, a temporary password has been sent' },
    { status: 200 },
);

export async function POST(request: Request) {
    try {
        const { email } = await request.json();

        // 1. Validate input
        if (!email) {
            return errorResponse('Email is required', 400, 'MISSING_FIELDS');
        }
        if (!EMAIL_REGEX.test(email)) {
            return errorResponse('A valid email address is required', 400, 'INVALID_EMAIL');
        }

        // 2. Look up user — return safe response if not found to avoid email enumeration
        const { data: userData, error: userError } = await supabase
            .from('user')
            .select('id, email')
            .eq('email', email)
            .single();

        if (userError || !userData) {
            return SAFE_RESPONSE;
        }

        // 3. Generate temporary password
        const tempPassword = generateTempPassword();

        // 4. Hash temporary password
        const hashedPassword = await bcrypt.hash(tempPassword, 10);

        // 5. Update user record with hashed temp password
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

        // 6. Send email with temporary password via SMTP
        const transporter = nodemailer.createTransport({
            host: process.env.SMTP_HOST,
            port: Number(process.env.SMTP_PORT),
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASSWORD,
            },
        });

        await transporter.sendMail({
            from: process.env.SMTP_FROM,
            to: email,
            subject: 'Your Temporary Password - MOP Platform',
            text:
                `Your temporary password is: ${tempPassword}\n\n` +
                `Please visit the following link to reset your password: ${process.env.NEXT_PUBLIC_APP_URL}/en/change-password?email=${encodeURIComponent(userData.email)}\n\n` +
                `This temporary password can only be used once.`,
        });

        // 7. Return safe response
        return SAFE_RESPONSE;
    } catch (error) {
        console.error('Forgot Password Error:', error);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
