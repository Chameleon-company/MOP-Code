/**
 * @jest-environment node
 *
 * Tests for PUT /api/profile/password
 * All Supabase and bcrypt calls are mocked — no real DB or crypto work happens here.
 */

// ─── Mocks ──────────────────────────────────────────────────────────────────

jest.mock('next/server', () => ({
    NextResponse: {
        json: jest
            .fn()
            .mockImplementation(
                (body: unknown, init?: { status?: number }) => ({
                    status: init?.status ?? 200,
                    json: jest.fn().mockResolvedValue(body),
                    _body: body,
                }),
            ),
    },
}));

jest.mock('@/library/supabaseClient', () => ({
    supabase: { from: jest.fn() },
}));

jest.mock('bcryptjs', () => ({
    compare: jest.fn(),
    hash: jest.fn(),
}));

// ─── Imports ─────────────────────────────────────────────────────────────────

import { PUT } from '../../../app/api/profile/password/route';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeRequest(body?: object, userId?: string) {
    const headers = new Map<string, string>();
    if (userId) headers.set('x-user-id', userId);
    return {
        headers: { get: (key: string) => headers.get(key) ?? null },
        json: jest.fn().mockResolvedValue(body ?? {}),
    } as any;
}

function makeChain(result: { data: unknown; error: unknown }) {
    const chain: Record<string, jest.Mock> = {};
    chain.select = jest.fn().mockReturnValue(chain);
    chain.eq = jest.fn().mockReturnValue(chain);
    chain.update = jest.fn().mockReturnValue(chain);
    chain.maybeSingle = jest.fn().mockResolvedValue(result);
    return chain;
}

const MOCK_USER = { password: '$2a$12$hashedpassword' };

const VALID_BODY = {
    current_password: 'OldPass123',
    new_password: 'NewPass456',
    confirm_password: 'NewPass456',
};

// ─── Tests: PUT /api/profile/password ────────────────────────────────────────

describe('PUT /api/profile/password', () => {
    beforeEach(() => jest.clearAllMocks());

    test('valid password change → 200 success', async () => {
        (supabase.from as jest.Mock).mockImplementation(() =>
            makeChain({ data: MOCK_USER, error: null }),
        );
        (bcrypt.compare as jest.Mock).mockResolvedValue(true);
        (bcrypt.hash as jest.Mock).mockResolvedValue('$2a$12$newhash');

        const res = await PUT(makeRequest(VALID_BODY, '9'));
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.message).toBe('Password updated successfully');
    });

    test('wrong current password → 401 Unauthorised', async () => {
        (supabase.from as jest.Mock).mockImplementation(() =>
            makeChain({ data: MOCK_USER, error: null }),
        );
        (bcrypt.compare as jest.Mock).mockResolvedValue(false);

        const res = await PUT(makeRequest(VALID_BODY, '9'));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.message).toBe('Current password is incorrect');
    });

    test('new_password and confirm_password do not match → 400 validation error', async () => {
        const res = await PUT(
            makeRequest(
                {
                    current_password: 'OldPass123',
                    new_password: 'NewPass456',
                    confirm_password: 'Different789',
                },
                '9',
            ),
        );
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('confirm_password');
    });

    test('new password same as current password → 400 validation error', async () => {
        const res = await PUT(
            makeRequest(
                {
                    current_password: 'SamePass123',
                    new_password: 'SamePass123',
                    confirm_password: 'SamePass123',
                },
                '9',
            ),
        );
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('new_password');
    });

    test('weak new password (no uppercase) → 400 validation error', async () => {
        const res = await PUT(
            makeRequest(
                {
                    current_password: 'OldPass123',
                    new_password: 'weakpass1',
                    confirm_password: 'weakpass1',
                },
                '9',
            ),
        );
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('new_password');
    });

    test('weak new password (too short) → 400 validation error', async () => {
        const res = await PUT(
            makeRequest(
                {
                    current_password: 'OldPass123',
                    new_password: 'Ab1',
                    confirm_password: 'Ab1',
                },
                '9',
            ),
        );
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('new_password');
    });

    test('missing current_password field → 400 validation error', async () => {
        const res = await PUT(
            makeRequest(
                {
                    new_password: 'NewPass456',
                    confirm_password: 'NewPass456',
                },
                '9',
            ),
        );
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('current_password');
    });

    test('no x-user-id header → 401 Unauthorised', async () => {
        const res = await PUT(makeRequest(VALID_BODY, undefined));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
    });

    test('user not found in DB → 401 Unauthorised', async () => {
        (supabase.from as jest.Mock).mockImplementation(() =>
            makeChain({ data: null, error: null }),
        );

        const res = await PUT(makeRequest(VALID_BODY, '9'));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
    });

    test('DB error on fetch → 500', async () => {
        (supabase.from as jest.Mock).mockImplementation(() =>
            makeChain({ data: null, error: { message: 'Connection refused' } }),
        );

        const res = await PUT(makeRequest(VALID_BODY, '9'));
        const body = await res.json();

        expect(res.status).toBe(500);
        expect(body.success).toBe(false);
    });
});
