import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import bcrypt from 'bcryptjs';
import nodemailer from 'nodemailer';
import { errorResponse } from '@/app/api/library/errorResponse';
import crypto from 'crypto';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const TEMP_PASSWORD_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
const TEMP_PASSWORD_LENGTH = 10;

const rateLimitCache = new Map<string, { count: number, resetTime: number }>();
const RATE_LIMIT_MAX_REQUESTS = 3;
const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes

function checkRateLimit(key: string): boolean {
    const now = Date.now();
    const record = rateLimitCache.get(key);
    
    if (!record) {
        rateLimitCache.set(key, { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS });
        return true;
    }
    
    if (now > record.resetTime) {
        rateLimitCache.set(key, { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS });
        return true;
    }
    
    if (record.count >= RATE_LIMIT_MAX_REQUESTS) {
        return false;
    }
    
    record.count++;
    return true;
}

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
        const ip = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown';
        if (ip !== 'unknown' && !checkRateLimit(`ip_${ip}`)) {
            return errorResponse('Too many requests, please try again later', 429, 'RATE_LIMIT_EXCEEDED');
        }

        const { email } = await request.json();

        // 1. Validate input
        if (!email || typeof email !== 'string') {
            return errorResponse('Email is required', 400, 'MISSING_FIELDS');
        }

        if (email.length > 254 || !EMAIL_REGEX.test(email)) {
            return errorResponse('A valid email address is required', 400, 'INVALID_EMAIL');
        }

        const normalizeEmail = email.toLowerCase().trim();

        if (!checkRateLimit(`email_${normalizeEmail}`)) {
            return errorResponse('Too many requests for this email, please try again later', 429, 'RATE_LIMIT_EXCEEDED');
        }

        await dbConnect();

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
            console.error('SMTP Error (swallowed for dev testing). Temp Password is:', tempPassword);
            console.error(emailError);
            // We intentionally don't throw here so developers can test the reset flow
            // by grabbing the temp password from the console.
        }

        return SAFE_RESPONSE;
    } catch (error) {
        console.error('Forgot Password Error:', error);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}