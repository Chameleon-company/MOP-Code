/**
 * @jest-environment node
 *
 * Tests for GET /api/profile and PUT /api/profile
 * All Supabase calls are mocked — no real DB work happens here.
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

// ─── Imports ─────────────────────────────────────────────────────────────────

import { GET, PUT } from '../../../app/api/profile/route';
import { supabase } from '@/library/supabaseClient';

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build a mock NextRequest with optional headers and body
function makeRequest(body?: object, userId?: string) {
    const headers = new Map<string, string>();
    if (userId) headers.set('x-user-id', userId);
    return {
        headers: { get: (key: string) => headers.get(key) ?? null },
        json: jest.fn().mockResolvedValue(body ?? {}),
    } as any;
}

// Fluent chain mock for Supabase queries that end with .single()
function makeChain(result: { data: unknown; error: unknown }) {
    const chain: Record<string, jest.Mock> = {};
    chain.select = jest.fn().mockReturnValue(chain);
    chain.eq = jest.fn().mockReturnValue(chain);
    chain.update = jest.fn().mockReturnValue(chain);
    chain.insert = jest.fn().mockReturnValue(chain);
    chain.single = jest.fn().mockResolvedValue(result);
    chain.maybeSingle = jest.fn().mockResolvedValue(result);
    return chain;
}

// Mock data
const MOCK_USER_DETAILS = {
    id: 2,
    user_id: 9,
    first_name: 'Jason',
    last_name: 'Holder',
    age: null,
    gender: null,
    profile_img: null,
    created_at: '2026-03-22T11:09:32.253182+00:00',
    updated_at: '2026-03-22T11:09:32.253182+00:00',
};
const MOCK_USER = { email: 'jason@gmail.com' };

// ─── Tests: GET /api/profile ──────────────────────────────────────────────────

describe('GET /api/profile', () => {
    beforeEach(() => jest.clearAllMocks());

    test('valid user → 200 with profile data and email', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details')
                return makeChain({ data: MOCK_USER_DETAILS, error: null });
            if (table === 'user')
                return makeChain({ data: MOCK_USER, error: null });
            return makeChain({ data: null, error: null });
        });

        const res = await GET(makeRequest(undefined, '9'));
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.first_name).toBe('Jason');
        expect(body.data.email).toBe('jason@gmail.com');
    });

    test('no x-user-id header → 401 Unauthorised', async () => {
        const res = await GET(makeRequest(undefined, undefined));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
    });

    test('user has no profile row yet → 200 with empty shell', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details')
                return makeChain({ data: null, error: null });
            if (table === 'user')
                return makeChain({ data: MOCK_USER, error: null });
            return makeChain({ data: null, error: null });
        });

        const res = await GET(makeRequest(undefined, '9'));
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.first_name).toBeNull();
        expect(body.data.email).toBe('jason@gmail.com');
    });

    test('DB error on user_details → 500', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details')
                return makeChain({
                    data: null,
                    error: { message: 'DB error' },
                });
            return makeChain({ data: null, error: null });
        });

        const res = await GET(makeRequest(undefined, '9'));
        const body = await res.json();

        expect(res.status).toBe(500);
        expect(body.success).toBe(false);
    });
});

// ─── Tests: PUT /api/profile ──────────────────────────────────────────────────

describe('PUT /api/profile', () => {
    beforeEach(() => jest.clearAllMocks());

    test('valid update of first_name and last_name → 200 success', async () => {
        const updatedData = {
            ...MOCK_USER_DETAILS,
            first_name: 'Jason',
            last_name: 'Smith',
        };

        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details') {
                const chain = makeChain({ data: updatedData, error: null });
                chain.maybeSingle = jest
                    .fn()
                    .mockResolvedValue({ data: { id: 2 }, error: null });
                return chain;
            }
            return makeChain({ data: null, error: null });
        });

        const res = await PUT(
            makeRequest({ first_name: 'Jason', last_name: 'Smith' }, '9'),
        );
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.message).toBe('Profile updated successfully');
    });

    test('valid update of email → 200 success', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user') return makeChain({ data: null, error: null });
            return makeChain({ data: null, error: null });
        });

        const res = await PUT(
            makeRequest({ email: 'newemail@gmail.com' }, '9'),
        );
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
    });

    test('valid update of age and gender → 200 success', async () => {
        const updatedData = { ...MOCK_USER_DETAILS, age: 25, gender: 'Male' };

        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details') {
                const chain = makeChain({ data: updatedData, error: null });
                chain.maybeSingle = jest
                    .fn()
                    .mockResolvedValue({ data: { id: 2 }, error: null });
                return chain;
            }
            return makeChain({ data: null, error: null });
        });

        const res = await PUT(makeRequest({ age: 25, gender: 'Male' }, '9'));
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
    });

    test('invalid gender value → 400 validation error', async () => {
        const res = await PUT(makeRequest({ gender: 'Batman' }, '9'));
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('gender');
    });

    test('age as string instead of number → 400 validation error', async () => {
        const res = await PUT(makeRequest({ age: '23' }, '9'));
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('age');
    });

    test('empty body → 400 no updatable fields', async () => {
        const res = await PUT(makeRequest({}, '9'));
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
    });

    test('no x-user-id header → 401 Unauthorised', async () => {
        const res = await PUT(makeRequest({ first_name: 'Jason' }, undefined));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
    });

    test('new user with no existing profile row → inserts and returns 200', async () => {
        const newData = {
            user_id: 99,
            first_name: 'New',
            last_name: 'User',
            age: null,
            gender: null,
            profile_img: null,
        };

        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user_details') {
                const chain = makeChain({ data: newData, error: null });
                chain.maybeSingle = jest
                    .fn()
                    .mockResolvedValue({ data: null, error: null }); // no existing row
                return chain;
            }
            return makeChain({ data: null, error: null });
        });

        const res = await PUT(
            makeRequest({ first_name: 'New', last_name: 'User' }, '99'),
        );
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
    });
});
