import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

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

        // Resolve the admin role_id from the roles table
        const { data: roleData, error: roleError } = await supabase
            .from('roles')
            .select('id')
            .eq('role_name', 'admin')
            .single();

        let adminEmails: string[] = [];

        if (roleError || !roleData) {
            console.warn('[contact] Could not resolve admin role:', roleError?.message);
        } else {
            // Fetch all users with the admin role_id
            const { data: adminUsers, error: usersError } = await supabase
                .from('user')
                .select('email')
                .eq('role_id', roleData.id);

            if (usersError || !adminUsers || adminUsers.length === 0) {
                console.warn('[contact] No admin users found:', usersError?.message);
            } else {
                adminEmails = adminUsers.map((u: { email: string }) => u.email);
                console.log('[contact] Admin emails to notify:', adminEmails);
            }
        }

        // Send notification email to each admin
        const emailBody =
            `New Contact Form Submission\n\n` +
            `From: ${fullName} <${email}>\n` +
            `Subject: ${subject}\n\n` +
            `Message:\n${message}`;

        for (const adminEmail of adminEmails) {
            try {
                await transporter.sendMail({
                    from: process.env.SMTP_FROM,
                    to: adminEmail,
                    subject: `Contact Form: ${subject}`,
                    text: emailBody,
                });
                console.log(`[contact] Notification sent to ${adminEmail}`);
            } catch (mailError) {
                console.error(`[contact] Failed to send email to ${adminEmail}:`, mailError);
            }
        }

        return NextResponse.json(
            { success: true, message: 'Message received' },
            { status: 200 },
        );
    } catch {
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
