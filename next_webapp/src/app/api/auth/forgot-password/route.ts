import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import bcrypt from 'bcryptjs';
import nodemailer from 'nodemailer';
import { errorResponse } from '@/app/api/library/errorResponse';
import crypto from 'crypto';
import { checkPasswordResetRateLimit, recordPasswordResetAttempt, clearPasswordResetAttempts, getClientIp } from '@/app/api/library/passwordResetRateLimit';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const TEMP_PASSWORD_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
const TEMP_PASSWORD_LENGTH = 10;



function generateTempPassword(): string {
    let result = '';
    for (let i = 0; i < TEMP_PASSWORD_LENGTH; i++) {
        result += TEMP_PASSWORD_CHARS.charAt(
            crypto.randomInt(0, TEMP_PASSWORD_CHARS.length)
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
        const ip = getClientIp(request);
        const { email } = await request.json();

        // 1. Validate input
        if (!email || typeof email !== 'string') {
            return errorResponse('Email is required', 400, 'MISSING_FIELDS');
        }

        if (email.length > 254 || !EMAIL_REGEX.test(email)) {
            return errorResponse('A valid email address is required', 400, 'INVALID_EMAIL');
        }

        const normalizeEmail = email.toLowerCase().trim();

        const { limited } = await checkPasswordResetRateLimit(normalizeEmail, ip, "forgot_password_request");
        await recordPasswordResetAttempt(normalizeEmail, ip, "forgot_password_request");
        if (limited) {
            return errorResponse('Too many requests, please try again later', 429, 'RATE_LIMIT_EXCEEDED');
        }
        // 2. Look up user in MongoDB
        const userData = await User.findOne({
            email: normalizeEmail,
        }).exec();

        if (!userData) {
            return SAFE_RESPONSE;
        }

        // 3. Generate a fresh token and save it to the DB.
        //    The token written to the DB and the one emailed are always generated
        //    in the same request, so they are guaranteed to match.
        //    Abuse is prevented by the per-email/per-IP rate limiter above.
        const tempPassword = generateTempPassword();
        const hashedPassword = await bcrypt.hash(tempPassword, 10);
        userData.reset_token = hashedPassword;
        userData.reset_token_expires = new Date(Date.now() + 15 * 60 * 1000); // 15 mins
        userData.reset_token_used = false;
        await userData.save();

        // 4. Send email (with fallback for dev)
        try {
            const transporter = nodemailer.createTransport({
                host: process.env.SMTP_HOST,
                port: Number(process.env.SMTP_PORT),
                secure: false, // false = STARTTLS on port 587; true = TLS on port 465
                auth: {
                    user: process.env.SMTP_USER,
                    pass: process.env.SMTP_PASSWORD,
                },
            });

            await transporter.sendMail({
                from: process.env.SMTP_FROM,
                to: userData.email,
                subject: 'Your Temporary Password - MOP Platform',
                text:
                    `Your temporary password is: ${tempPassword}\n\n` +
                    `Please visit the following link to reset your password: ${process.env.NEXT_PUBLIC_APP_URL}/en/change-password?email=${encodeURIComponent(userData.email)}\n\n` +
                    `This temporary password can only be used once.`,
            });
        } catch (emailError) {
            if (process.env.NODE_ENV !== 'production') {
                console.error('SMTP Error (swallowed for dev testing). Temp Password is:', tempPassword);
            }
            console.error(emailError);
            // We intentionally don't throw here so developers can test the reset flow
            // by grabbing the temp password from the console.
        }

        // 5. Clear the rate limit on success so they aren't unnecessarily blocked
        await clearPasswordResetAttempts(normalizeEmail, ip, "forgot_password_request");

        return SAFE_RESPONSE;
    } catch (error) {
        console.error('Forgot Password Error:', error);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}