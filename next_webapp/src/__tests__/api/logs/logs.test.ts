/**
 * @jest-environment node
 *
 * Tests for GET /api/logs
 * All Supabase calls and logger calls are mocked — no real DB or I/O work here.
 */

// ─── Mocks ──────────────────────────────────────────────────────────────────

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

jest.mock('@/app/api/library/auth', () => ({
  getAuthUser: jest.fn(),
}));

jest.mock('@/utils/logger', () => ({
  __esModule: true,
  default: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

// ─── Imports ─────────────────────────────────────────────────────────────────

import { GET } from '../../../app/api/logs/route';
import { supabase } from '@/library/supabaseClient';
import { getAuthUser } from '@/app/api/library/auth';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeRequest(url = 'http://localhost:3000/api/logs') {
  return { url, headers: { get: () => null } } as any;
}

// The logs route applies filters AFTER .range(), so .range() must return the
// chain itself (not a Promise). The chain is made directly awaitable via .then.
function makeChain(result: { data: unknown; error: unknown; count?: number | null }) {
  const chain: Record<string, jest.Mock> = {};
  chain.select = jest.fn().mockReturnValue(chain);
  chain.order  = jest.fn().mockReturnValue(chain);
  chain.range  = jest.fn().mockReturnValue(chain);
  chain.eq     = jest.fn().mockReturnValue(chain);
  chain.gte    = jest.fn().mockReturnValue(chain);
  chain.lte    = jest.fn().mockReturnValue(chain);
  chain.ilike  = jest.fn().mockReturnValue(chain);
  const resolvedPromise = Promise.resolve(result);
  Object.defineProperty(chain, 'then', {
    get: () => resolvedPromise.then.bind(resolvedPromise),
  });
  return chain;
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

const ADMIN_AUTH = { userId: 9, roleId: 1, roleName: 'admin', isAuthenticated: true,  isAdmin: true  };
const USER_AUTH  = { userId: 7, roleId: 2, roleName: 'user',  isAuthenticated: true,  isAdmin: false };
const ANON_AUTH  = { userId: null, roleId: null, roleName: null, isAuthenticated: false, isAdmin: false };

const MOCK_LOGS = [
  { id: 1, level: 'info',  message: 'Request logged', source: 'middleware', timestamp: '2026-01-01T00:00:00.000Z' },
  { id: 2, level: 'error', message: 'DB error',        source: 'api',        timestamp: '2026-01-01T00:01:00.000Z' },
];

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('GET /api/logs', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
    (supabase.from as jest.Mock).mockReturnValue(
      makeChain({ data: MOCK_LOGS, error: null, count: 2 }),
    );
  });

  test('unauthenticated request → 401', async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);
    const res = await GET(makeRequest());
    const body = await res.json();
    expect(res.status).toBe(401);
    expect(body.success).toBe(false);
  });

  test('authenticated non-admin → 403', async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);
    const res = await GET(makeRequest());
    const body = await res.json();
    expect(res.status).toBe(403);
    expect(body.success).toBe(false);
  });

  test('admin → 200 with correct data and full pagination shape', async () => {
    const res = await GET(makeRequest());
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual(MOCK_LOGS);
    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 50,
      total: 2,
      totalPages: 1,
      hasNext: false,
      hasPrev: false,
    });
  });

  test('level filter → .eq("level", value) is applied to the query', async () => {
    const chain = makeChain({ data: [MOCK_LOGS[1]], error: null, count: 1 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost:3000/api/logs?level=error'));
    expect(res.status).toBe(200);
    expect(chain.eq).toHaveBeenCalledWith('level', 'error');
  });

  test('user_id filter → .eq("user_id", integer) is applied to the query', async () => {
    const chain = makeChain({ data: [MOCK_LOGS[0]], error: null, count: 1 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost:3000/api/logs?user_id=42'));
    expect(res.status).toBe(200);
    expect(chain.eq).toHaveBeenCalledWith('user_id', 42);
  });

  test('Supabase failure → 500 with success: false', async () => {
    (supabase.from as jest.Mock).mockReturnValue(
      makeChain({ data: null, error: { message: 'Connection refused' }, count: null }),
    );
    const res = await GET(makeRequest());
    const body = await res.json();
    expect(res.status).toBe(500);
    expect(body.success).toBe(false);
  });
});
