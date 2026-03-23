/**
 * @jest-environment node
 *
 * Tests for POST /api/auth/login
 * All Supabase, bcrypt and jsonwebtoken calls are mocked — no real DB or
 * crypto work happens here.
 */

// Mocks (must be declared before any imports that pull in the mocked modules) 

jest.mock('next/server', () => ({
    NextResponse: {
        json: jest.fn().mockImplementation((body: unknown, init?: { status?: number }) => ({
            status: init?.status ?? 200,
            json: jest.fn().mockResolvedValue(body),
            _body: body,
        })),
    },
}));

jest.mock('@/library/supabaseClient', () => ({
    supabase: { from: jest.fn() },
}));

jest.mock('bcryptjs', () => ({
    compare: jest.fn(),
}));

jest.mock('jsonwebtoken', () => ({
    sign: jest.fn(),
}));

// Imports

import { POST } from '../../../app/api/auth/login/route';
import { supabase } from '@/library/supabaseClient';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// Helpers

// Build a minimal mock Request with a JSON body.
function makeRequest(body: object) {
    return { json: jest.fn().mockResolvedValue(body) } as unknown as Request;
}

//Fluent-chain mock that returns a fixed value from `.single()`.
function makeChain(result: { data: unknown; error: unknown }) {
    const chain: Record<string, jest.Mock> = {};
    chain.select = jest.fn().mockReturnValue(chain);
    chain.eq     = jest.fn().mockReturnValue(chain);
    chain.single = jest.fn().mockResolvedValue(result);
    return chain;
}

const MOCK_USER     = { id: 1, email: 'user@test.com', password: '$2a$10$hash', role_id: 1 };
const MOCK_ROLE     = { id: 1, role_name: 'user' };
const MOCK_DETAILS  = { user_id: 1, first_name: 'Jane', last_name: 'Doe' };
const MOCK_TOKEN    = 'mock.jwt.token';

//Wire Supabase for the happy path (all three queries succeed).
function setupHappyPath() {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
        if (table === 'user')         return makeChain({ data: MOCK_USER,    error: null });
        if (table === 'roles')        return makeChain({ data: MOCK_ROLE,    error: null });
        if (table === 'user_details') return makeChain({ data: MOCK_DETAILS, error: null });
        return makeChain({ data: null, error: null });
    });
    (bcrypt.compare as jest.Mock).mockResolvedValue(true);
    (jwt.sign as jest.Mock).mockReturnValue(MOCK_TOKEN);
}

// Tests

beforeEach(() => {
    jest.clearAllMocks();
    process.env.JWT_SECRET = 'test-secret';
});

describe('POST /api/auth/login', () => {
    test('valid credentials → 200 with token', async () => {
        setupHappyPath();
        const res = await POST(makeRequest({ email: 'user@test.com', password: 'password123' }));
        const body = await res.json();

        expect(res.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.token).toBe(MOCK_TOKEN);
        expect(body.data.email).toBe(MOCK_USER.email);
        expect(body.data.firstName).toBe(MOCK_DETAILS.first_name);
    });

    test('wrong password → 401 INVALID_CREDENTIALS', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'user') return makeChain({ data: MOCK_USER, error: null });
            return makeChain({ data: null, error: null });
        });
        (bcrypt.compare as jest.Mock).mockResolvedValue(false);

        const res = await POST(makeRequest({ email: 'user@test.com', password: 'wrongpass' }));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INVALID_CREDENTIALS');
    });

    test('unknown email → 401 INVALID_CREDENTIALS', async () => {
        (supabase.from as jest.Mock).mockImplementation(() =>
            makeChain({ data: null, error: { message: 'Row not found' } })
        );

        const res = await POST(makeRequest({ email: 'nobody@test.com', password: 'password123' }));
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INVALID_CREDENTIALS');
    });

    test('missing fields → 400 MISSING_FIELDS', async () => {
        const res = await POST(makeRequest({ email: 'user@test.com' })); // no password
        const body = await res.json();

        expect(res.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.code).toBe('MISSING_FIELDS');
        expect(supabase.from).not.toHaveBeenCalled();
    });

    test('DB throws unexpectedly → 500 INTERNAL_ERROR', async () => {
        (supabase.from as jest.Mock).mockImplementation(() => {
            throw new Error('Connection refused');
        });

        const res = await POST(makeRequest({ email: 'user@test.com', password: 'password123' }));
        const body = await res.json();

        expect(res.status).toBe(500);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INTERNAL_ERROR');
    });
});
