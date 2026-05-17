import { NextResponse } from 'next/server';
import logger from '../../../utils/logger';

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
    request?: Request,
    userId?: number
): NextResponse<ErrorBody> {
    // Log the error with additional context
    logger.error(`API Error Response: ${status} - ${message}${code ? ` (${code})` : ''}`, {
        source: 'api',
        status_code: status,
        error_code: code,
        user_id: userId,
        url: request?.url,
        method: request?.method,
        ip_address: request?.headers?.get('x-forwarded-for') || request?.headers?.get('x-real-ip'),
        user_agent: request?.headers?.get('user-agent'),
    });

    const body: ErrorBody = { success: false, message };
    if (code !== undefined) body.code = code;
    return NextResponse.json(body, { status });
}
