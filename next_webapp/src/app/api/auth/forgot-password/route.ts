import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
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
        if (!email || typeof email !== 'string') {
            return errorResponse('Email is required', 400, 'MISSING_FIELDS');
        }

        if (email.length > 254 || !EMAIL_REGEX.test(email)) {
            return errorResponse('A valid email address is required', 400, 'INVALID_EMAIL');
        }

        await dbConnect();

        const normalizeEmail = email.toLowerCase().trim();

        // 2. Look up user in MongoDB
        const userData = await User.findOne({
            email: normalizeEmail,
        }).exec();

        if (!userData) {
            return SAFE_RESPONSE;
        }

        // 3. Generate temporary password
        const tempPassword = generateTempPassword();

        // 4. Hash temporary password
        const hashedPassword = await bcrypt.hash(tempPassword, 10);

        // 5. Update MongoDB user
        userData.password = hashedPassword;
        await userData.save();

        // 6. Send email
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
            to: userData.email,
            subject: 'Your Temporary Password - MOP Platform',
            text:
                `Your temporary password is: ${tempPassword}\n\n` +
                `Please visit the following link to reset your password: ${process.env.NEXT_PUBLIC_APP_URL}/en/change-password?email=${encodeURIComponent(userData.email)}\n\n` +
                `This temporary password can only be used once.`,
        });

        return SAFE_RESPONSE;
    } catch (error) {
        console.error('Forgot Password Error:', error);
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}