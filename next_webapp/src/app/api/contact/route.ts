import { NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
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

        // adminEmails is ready for the email-sending task (Task 3)

        return NextResponse.json(
            { success: true, message: 'Message received' },
            { status: 200 },
        );
    } catch {
        return errorResponse('Internal Server Error', 500, 'INTERNAL_ERROR');
    }
}
