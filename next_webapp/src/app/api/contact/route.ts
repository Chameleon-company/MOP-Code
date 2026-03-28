import { NextResponse } from 'next/server';
import { errorResponse } from '@/app/api/library/errorResponse';

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

        return NextResponse.json(
            { success: true, message: 'Message received' },
            { status: 200 },
        );
    } catch {
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
