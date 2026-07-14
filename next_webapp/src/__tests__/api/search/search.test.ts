/**
 * @jest-environment node
 *
 * Tests for GET /api/search
 * All Supabase calls and errorResponse are mocked — no real DB calls happen.
 */

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

import { GET } from '../../../app/api/search/route';
import { supabase } from '@/library/supabaseClient';
import { errorResponse } from '@/app/api/library/errorResponse';

// Helpers

function makeRequest(url: string) {
  return { url } as unknown as Request;
}

// Returns a fluent chain that is also directly awaitable (resolves to result).
// .single() resolves to result too, for queries that use it.
function makeChain(result: { data: unknown; error: unknown }) {
  const chain: Record<string, any> = {
    then(onFulfilled: any, onRejected: any) {
      return Promise.resolve(result).then(onFulfilled, onRejected);
    },
    single: jest.fn().mockResolvedValue(result),
  };
  chain.select = jest.fn().mockReturnValue(chain);
  chain.ilike  = jest.fn().mockReturnValue(chain);
  chain.eq     = jest.fn().mockReturnValue(chain);
  return chain;
}

// Fixtures

const UC_1 = { id: 1, title: 'ML Basics',       description: 'Intro to machine learning',   category_id: 1 };
const UC_2 = { id: 2, title: 'Deep Learning',    description: 'Neural network fundamentals', category_id: 3 };
const UC_3 = { id: 3, title: 'Climate Forecast', description: 'Weather prediction models',   category_id: 2 };

beforeEach(() => {
  jest.clearAllMocks();
});

describe('GET /api/search', () => {
  test('no params — returns all use cases', async () => {
    (supabase.from as jest.Mock).mockReturnValue(
      makeChain({ data: [UC_1, UC_2, UC_3], error: null }),
    );

    const res = await GET(makeRequest('http://localhost/api/search'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data.results).toHaveLength(3);
    expect(body.data.total).toBe(3);
    expect(body.data.filters).toEqual({ q: null, title: null, category: null, tag: null });
  });

  test('keyword search (q) — returns deduplicated results matching title or description', async () => {
    // UC_1 appears in both title and description results — should be deduped to one entry
    (supabase.from as jest.Mock)
      .mockReturnValueOnce(makeChain({ data: [UC_1],         error: null })) // title ilike
      .mockReturnValueOnce(makeChain({ data: [UC_1, UC_2],   error: null })); // desc ilike

    const res = await GET(makeRequest('http://localhost/api/search?q=ml'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(2);
    expect(body.data.results.map((r: any) => r.id)).toEqual([1, 2]);
    expect(body.data.total).toBe(2);
  });

  test('title search — returns results matching title only', async () => {
    (supabase.from as jest.Mock).mockReturnValue(
      makeChain({ data: [UC_2], error: null }),
    );

    const res = await GET(makeRequest('http://localhost/api/search?title=deep'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    expect(body.data.results[0].id).toBe(2);
    // Only one DB call — title branch does a single query, not two
    expect(supabase.from).toHaveBeenCalledTimes(1);
  });

  test('q takes priority over title when both provided', async () => {
    (supabase.from as jest.Mock)
      .mockReturnValueOnce(makeChain({ data: [UC_1], error: null })) // q title ilike
      .mockReturnValueOnce(makeChain({ data: [UC_3], error: null })); // q desc ilike

    const res = await GET(makeRequest('http://localhost/api/search?q=ml&title=deep'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results.map((r: any) => r.id)).toEqual([1, 3]);
    // Exactly 2 calls — q branch only, no extra title-filter call
    expect(supabase.from).toHaveBeenCalledTimes(2);
  });

  test('category filter — filters results by category_id', async () => {
    (supabase.from as jest.Mock).mockReturnValue(
      makeChain({ data: [UC_2], error: null }),
    );

    const res = await GET(makeRequest('http://localhost/api/search?category=3'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    expect(body.data.results[0].category_id).toBe(3);
  });

  test('invalid category — returns 400 INVALID_CATEGORY', async () => {
    await GET(makeRequest('http://localhost/api/search?category=notanumber'));

    expect(errorResponse).toHaveBeenCalledWith(
      'category must be a valid integer',
      400,
      'INVALID_CATEGORY',
    );
    expect(supabase.from).not.toHaveBeenCalled();
  });

  test('tag search — looks up tag by slug, filters by usecase_ids', async () => {
    (supabase.from as jest.Mock)
      .mockReturnValueOnce(makeChain({ data: [UC_1, UC_2, UC_3], error: null }))              // fetch all
      .mockReturnValueOnce(makeChain({ data: { id: 10 },          error: null }))              // tags lookup
      .mockReturnValueOnce(makeChain({ data: [{ usecase_id: 1 }, { usecase_id: 3 }], error: null })); // usecase_tags

    const res = await GET(makeRequest('http://localhost/api/search?tag=ml'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results.map((r: any) => r.id)).toEqual([1, 3]);
    expect(body.data.total).toBe(2);
  });

  test('unknown tag slug — returns empty results with total 0', async () => {
    (supabase.from as jest.Mock)
      .mockReturnValueOnce(makeChain({ data: [UC_1, UC_2], error: null }))                   // fetch all
      .mockReturnValueOnce(makeChain({ data: null, error: { message: 'Row not found' } })); // tags lookup — not found

    const res = await GET(makeRequest('http://localhost/api/search?tag=ghost-tag'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(0);
    expect(body.data.total).toBe(0);
  });

  test('keyword + category combined — applies both filters', async () => {
    (supabase.from as jest.Mock)
      .mockReturnValueOnce(makeChain({ data: [UC_2], error: null })) // title ilike + category eq
      .mockReturnValueOnce(makeChain({ data: [],     error: null })); // desc ilike + category eq

    const res = await GET(makeRequest('http://localhost/api/search?q=deep&category=3'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    expect(body.data.results[0].id).toBe(2);
  });

  test('internal error — returns 500 INTERNAL_ERROR', async () => {
    (supabase.from as jest.Mock).mockImplementation(() => {
      throw new Error('DB connection failed');
    });

    await GET(makeRequest('http://localhost/api/search'));

    expect(errorResponse).toHaveBeenCalledWith('Internal server error', 500, 'INTERNAL_ERROR');
  });
});
