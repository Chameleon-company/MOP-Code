/**
 * @jest-environment node
 *
 * Tests for GET /api/usecases — pagination behavior.
 * All Supabase calls are mocked — no real DB work happens here.
 */

// ─── Mocks ───────────────────────────────────────────────────────────────────

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

jest.mock('@/app/api/library/errorResponse', () => ({
  errorResponse: jest.fn().mockImplementation((message: string, status: number, code?: string) => ({
    status,
    json: jest.fn().mockResolvedValue({ success: false, message, code }),
    _body: { success: false, message, code },
  })),
}));

// ─── Imports ─────────────────────────────────────────────────────────────────

import { GET } from '../../../app/api/usecases/route';
import { supabase } from '@/library/supabaseClient';

const jestExpect = globalThis.expect as unknown as jest.Expect;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeRequest(url: string) {
  return { url } as unknown as Request;
}

// Fluent chain: all methods return the chain; .range() and .single() resolve.
// The chain itself is also directly awaitable via .then (for queries like .select().in()).
function makeChain(result: { data: unknown; error: unknown; count?: number }) {
  const chain: Record<string, any> = {};
  chain.select = jest.fn().mockReturnValue(chain);
  chain.or     = jest.fn().mockReturnValue(chain);
  chain.eq     = jest.fn().mockReturnValue(chain);
  chain.in     = jest.fn().mockReturnValue(chain);
  chain.range  = jest.fn().mockResolvedValue(result);
  chain.single = jest.fn().mockResolvedValue(result);
  chain.then   = (resolve: any, reject: any) => Promise.resolve(result).then(resolve, reject);
  chain.ilike  = jest.fn().mockReturnValue(chain);
  chain.order  = jest.fn().mockReturnValue(chain);
  return chain;
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

const UC_1 = { id: 1, title: 'ML Basics',    description: 'Intro to ML',    category_id: 1 };
const UC_2 = { id: 2, title: 'Deep Learning', description: 'Neural nets',    category_id: 3 };
const UC_3 = { id: 3, title: 'Climate',       description: 'Weather models', category_id: 2 };

beforeEach(() => {
  jest.clearAllMocks();
});

// ─── Tests: GET /api/usecases - pagination ───────────────────────────────────

describe('GET /api/usecases - pagination', () => {

  test('no params → defaults page=1 pageSize=10, calls range(0,9), returns metadata', async () => {
    const chain = makeChain({ data: [UC_1, UC_2, UC_3], error: null, count: 3 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.pagination).toEqual({ page: 1, pageSize: 10, total: 3, totalPages: 1 });
    jestExpect(chain.range).toHaveBeenCalledWith(0, 9);
  });

  test('page=2 pageSize=5 → calls range(5,9) and metadata reflects request', async () => {
    const chain = makeChain({ data: [UC_3], error: null, count: 11 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?page=2&pageSize=5'));
    const body = await res.json();

    jestExpect(body.pagination).toEqual({ page: 2, pageSize: 5, total: 11, totalPages: 3 });
    jestExpect(chain.range).toHaveBeenCalledWith(5, 9);
  });

  test('totalPages rounds up (count=7, pageSize=3 → 3 pages)', async () => {
    const chain = makeChain({ data: [UC_1, UC_2, UC_3], error: null, count: 7 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?pageSize=3'));
    const body = await res.json();

    jestExpect(body.pagination.totalPages).toBe(3);
  });

  test('invalid page param → returns 400 INVALID_PAGE', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?page=abc'));
    const body = await res.json();

    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_PAGE');
    jestExpect(chain.range).not.toHaveBeenCalled();
  });

  test('no results → total=0 totalPages=0', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases'));
    const body = await res.json();

    jestExpect(body.pagination).toEqual({ page: 1, pageSize: 10, total: 0, totalPages: 0 });
  });

  test('count field on response equals length of current page data', async () => {
    const chain = makeChain({ data: [UC_1, UC_2], error: null, count: 50 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?pageSize=2'));
    const body = await res.json();

    jestExpect(body.count).toBe(2);
    jestExpect(body.pagination.total).toBe(50);
  });

});

// ─── Tests: early-return paths include pagination metadata ────────────────────

describe('GET /api/usecases - tag filter early returns', () => {

  test('unknown tag slug → empty response with pagination metadata', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: null, error: { message: 'Not found' } });
      return makeChain({ data: [], error: null, count: 0 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag=ghost'));
    const body = await res.json();

    jestExpect(body.success).toBe(true);
    jestExpect(body.data).toHaveLength(0);
    jestExpect(body.pagination).toEqual({ page: 1, pageSize: 10, total: 0, totalPages: 0 });
  });

  test('unknown tag slug with custom page/pageSize → metadata echoes those params', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: null, error: { message: 'Not found' } });
      return makeChain({ data: [], error: null, count: 0 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag=ghost&page=3&pageSize=5'));
    const body = await res.json();

    jestExpect(body.pagination).toEqual({ page: 3, pageSize: 5, total: 0, totalPages: 0 });
  });

  test('tag found but no usecases linked → empty response with pagination metadata', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: { id: 5 }, error: null });
      if (table === 'usecase_tags') return makeChain({ data: [], error: null });
      return makeChain({ data: [], error: null, count: 0 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag=rare-tag'));
    const body = await res.json();

    jestExpect(body.success).toBe(true);
    jestExpect(body.data).toHaveLength(0);
    jestExpect(body.pagination.total).toBe(0);
  });

  test('tag filter with matches → paginated results include metadata', async () => {
    const usecasesChain = makeChain({ data: [UC_1, UC_2], error: null, count: 2 });

    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: { id: 7 }, error: null });
      if (table === 'usecase_tags') {
        return makeChain({ data: [{ usecase_id: 1 }, { usecase_id: 2 }], error: null });
      }
      return usecasesChain;
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag=ml&pageSize=5'));
    const body = await res.json();

    jestExpect(body.success).toBe(true);
    jestExpect(body.pagination.pageSize).toBe(5);
    jestExpect(body.pagination.total).toBe(2);
    jestExpect(body.pagination.totalPages).toBe(1);
    jestExpect(usecasesChain.range).toHaveBeenCalledWith(0, 4);
  });

});
