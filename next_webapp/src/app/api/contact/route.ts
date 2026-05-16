import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';
import { errorResponse } from '@/app/api/library/errorResponse';

const CONTACT_RECIPIENT = 'mopchameleon@gmail.com';

const REQUIRED_SMTP_VARS = [
    'SMTP_HOST',
    'SMTP_PORT',
    'SMTP_USER',
    'SMTP_PASSWORD',
    'SMTP_FROM',
] as const;

for (const varName of REQUIRED_SMTP_VARS) {
    if (!process.env[varName]) {
        throw new Error(`Missing required environment variable: ${varName}`);
    }
}

const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT),
    secure: false, // STARTTLS on port 587
    auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASSWORD,
    },
});

export async function POST(request: Request) {
    try {
        const { fullName, email, subject, message } = await request.json();

        if (!fullName || typeof fullName !== 'string' || fullName.trim() === '') {
            return errorResponse('Full name is required', 400, 'MISSING_FULL_NAME');
        }

        if (!email || typeof email !== 'string' || !/^\S+@\S+\.\S+$/.test(email)) {
            return errorResponse('A valid email address is required', 400, 'INVALID_EMAIL');
        }

        if (!subject || typeof subject !== 'string' || subject.trim() === '') {
            return errorResponse('Subject is required', 400, 'MISSING_SUBJECT');
        }

        if (!message || typeof message !== 'string' || message.trim() === '') {
            return errorResponse('Message is required', 400, 'MISSING_MESSAGE');
        }

        if (fullName.length > 100) {
            return errorResponse('Full name too long', 400, 'INVALID_FULL_NAME');
        }

        if (subject.length > 150) {
            return errorResponse('Subject too long', 400, 'INVALID_SUBJECT');
        }

        if (message.length > 2000) {
            return errorResponse('Message too long', 400, 'INVALID_MESSAGE');
        }

        const emailBody =
            `New Contact Form Submission\n\n` +
            `From: ${fullName} <${email}>\n` +
            `Subject: ${subject}\n\n` +
            `Message:\n${message}`;

        try {
            await transporter.sendMail({
                from: process.env.SMTP_FROM,
                to: CONTACT_RECIPIENT,
                subject: `Contact Form: ${subject}`,
                text: emailBody,
            });
            console.log(`[contact] Notification sent to ${CONTACT_RECIPIENT}`);
        } catch (mailError) {
            console.error(`[contact] Failed to send email to ${CONTACT_RECIPIENT}:`, mailError);
            return errorResponse(
                'Failed to send notification email',
                500,
                'EMAIL_FAILED'
            );
        }

        return NextResponse.json(
            { success: true, message: 'Message received' },
            { status: 200 },
        );
    } catch (error: any) {
        console.error('[contact] Unexpected error:', error);

        return errorResponse(
            'Internal Server Error',
            500,
            'INTERNAL_ERROR'
        );
    }
}
