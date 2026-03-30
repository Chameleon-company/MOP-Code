import { NextResponse } from 'next/server';

interface ErrorBody {
    success: false;
    message: string;
    code?: string;
}

/**
 * Return a consistent error JSON response across all auth API routes.
 *
 * @param message 
 * @param status  
 * @param code    
 */
export function errorResponse(
    message: string,
    status: number,
    code?: string,
): NextResponse<ErrorBody> {
    const body: ErrorBody = { success: false, message };
    if (code !== undefined) body.code = code;
    return NextResponse.json(body, { status });
}
