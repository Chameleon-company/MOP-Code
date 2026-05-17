/**
 * @jest-environment node
 *
 * Tests for POST /api/categories
 * Verifies that only admin users can create categories (RBAC).
 * All Supabase calls are mocked — no real DB work happens here.
 */

// ==============================
// Mocks — must come before imports
// ==============================

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

// ==============================
// Imports
// ==============================

import { POST } from '../../../app/api/categories/route';
import { supabase } from '@/library/supabaseClient';

// ==============================
// Helpers
// ==============================

/**
 * Builds a mock NextRequest with the given roleId in headers and a JSON body.
 * - roleId: 1       → admin user
 * - roleId: 2       → regular user
 * - roleId: null    → unauthenticated (no token)
 */
function makeRequest(roleId: number | null, body: object = {}) {
    return {
        headers: {
            get: (key: string) => {
                if (key === 'x-user-id')      return roleId !== null ? '1' : null;
                if (key === 'x-user-role-id') return roleId !== null ? String(roleId) : null;
                if (key === 'x-user-role')    return roleId === 1 ? 'admin' : 'user';
                return null;
            },
        },
        json: jest.fn().mockResolvedValue(body),
    } as any;
}

/**
 * Fluent-chain mock for Supabase queries.
 * Covers: select, eq, ilike, maybeSingle, insert, single
 */
function makeChain(result: { data: unknown; error: unknown }) {
    const chain: Record<string, jest.Mock> = {};
    chain.select      = jest.fn().mockReturnValue(chain);
    chain.eq          = jest.fn().mockReturnValue(chain);
    chain.ilike       = jest.fn().mockReturnValue(chain);
    chain.maybySingle = jest.fn().mockResolvedValue({ data: null, error: null });
    chain.maybeSingle = jest.fn().mockResolvedValue({ data: null, error: null });
    chain.insert      = jest.fn().mockReturnValue(chain);
    chain.single      = jest.fn().mockResolvedValue(result);
    return chain;
}

// ==============================
// Mock Data
// ==============================

const MOCK_CATEGORY = {
    id: 1,
    category_name: 'Test Category',
    description: 'A test category',
    created_by: 1,
};

const MOCK_USER = {
    id: 1,
    email: 'admin@test.com',
    role_id: 1,
};

// ==============================
// Setup
// ==============================

beforeEach(() => {
    jest.clearAllMocks();
});

// ==============================
// Tests
// ==============================

describe('POST /api/categories — RBAC', () => {

    test('admin user → 201 category created', async () => {
        (supabase.from as jest.Mock).mockImplementation((table: string) => {
            if (table === 'categories') return makeChain({ data: MOCK_CATEGORY, error: null });
            if (table === 'user')       return makeChain({ data: MOCK_USER, error: null });
            return makeChain({ data: null, error: null });
        });

        const res = await POST(
            makeRequest(1, { category_name: 'Test Category', description: 'A test category' })
        );
        const body = await res.json();

        expect(res.status).toBe(201);
        expect(body.success).toBe(true);
        expect(body.message).toBe('Category created successfully');
    });

    test('regular user → 403 FORBIDDEN', async () => {
        const res = await POST(
            makeRequest(2, { category_name: 'Test Category' })
        );
        const body = await res.json();

        expect(res.status).toBe(403);
        expect(body.success).toBe(false);
        expect(body.code).toBe('FORBIDDEN');
        // DB should never be touched — auth check happens first
        expect(supabase.from).not.toHaveBeenCalled();
    });

    test('unauthenticated (no token) → 401 UNAUTHORIZED', async () => {
        const res = await POST(
            makeRequest(null, { category_name: 'Test Category' })
        );
        const body = await res.json();

        expect(res.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.code).toBe('UNAUTHORIZED');
        // DB should never be touched — auth check happens first
        expect(supabase.from).not.toHaveBeenCalled();
    });

});