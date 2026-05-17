/**
 * @jest-environment node
 *
 * Tests for POST /api/auth/logout
 * The endpoint is stateless — no mocks beyond next/server are required.
 */

// Mocks

jest.mock('next/server', () => ({
    NextResponse: {
        json: jest.fn().mockImplementation((body: unknown, init?: { status?: number }) => ({
            status: init?.status ?? 200,
            json: jest.fn().mockResolvedValue(body),
        })),
    },
}));

// Imports 

import { POST } from '../../../app/api/auth/logout/route';

// Tests 

describe('POST /api/auth/logout', () => {
    test('returns 200 with success message', async () => {
        const res = await POST();
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.message).toBe('Logged out successfully');
    });
});
