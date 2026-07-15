import { NextResponse } from 'next/server';

export async function POST() {
    // Stateless logout: client discards the token; server-side expiration is 7 days.
    //
    // Unified logout target to maintain a stable client API and allow for FUTURE server-side session blacklisting.
    return NextResponse.json(
        { success: true, message: 'Logged out successfully' },
        { status: 200 },
    );
}
