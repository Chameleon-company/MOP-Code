/**
 * @jest-environment node
 *
 * Tests for GET /api/usecases — query parameter validation and filter combinations.
 * Covers:
 * - Validate query parameters (invalid types, edge cases)
 * - Test filter combinations (keyword + category, category + tags, all combined)
 *
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

jest.mock('../../../app/api/library/errorResponse', () => ({
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

function makeChain(result: { data: unknown; error: unknown; count?: number }) {
  const chain: Record<string, any> = {};
  chain.select = jest.fn().mockReturnValue(chain);
  chain.or     = jest.fn().mockReturnValue(chain);
  chain.eq     = jest.fn().mockReturnValue(chain);
  chain.in     = jest.fn().mockReturnValue(chain);
  chain.ilike  = jest.fn().mockReturnValue(chain);
  chain.order  = jest.fn().mockReturnValue(chain);
  chain.range  = jest.fn().mockResolvedValue(result);
  chain.single = jest.fn().mockResolvedValue(result);
  chain.then   = (resolve: any, reject: any) => Promise.resolve(result).then(resolve, reject);
  return chain;
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

const UC_1 = { id: 1, title: 'Playground A', description: 'Kids park',    category_id: 1 };
const UC_2 = { id: 2, title: 'Playground B', description: 'Water park',   category_id: 1 };
const UC_3 = { id: 3, title: 'Tech Hub',     description: 'Tech space',   category_id: 2 };

beforeEach(() => {
  jest.clearAllMocks();
});

// ─── Tests: Query Parameter Validation ───────────────────────────────────────

describe('GET /api/usecases - query parameter validation', () => {

  test('invalid category_id (non-numeric) → 400 INVALID_CATEGORY_ID', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?category_id=abc'));
    const body = await res.json();

    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_CATEGORY_ID');
  });

  test('invalid tag_id (non-numeric) → 400 INVALID_TAG_ID', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?tag_id=xyz'));
    const body = await res.json();

    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_TAG_ID');
  });

  test('invalid tag_ids (all non-numeric) → 400 INVALID_TAG_IDS', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?tag_ids=abc,def'));
    const body = await res.json();

    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_TAG_IDS');
  });

  test('valid tag_ids with mixed valid/invalid → filters invalid, uses valid ones', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') return makeChain({ data: [{ usecase_id: 1 }], error: null });
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    // "2,abc,4" — only 2 and 4 are valid numbers, abc gets filtered out
    const res = await GET(makeRequest('http://localhost/api/usecases?tag_ids=2,abc,4'));
    const body = await res.json();

    // Should succeed with the valid IDs
    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('valid category_id → applies filter and returns 200', async () => {
    const chain = makeChain({ data: [UC_1, UC_2], error: null, count: 2 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?category_id=1'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('valid tag_id → applies filter and returns 200', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') return makeChain({ data: [{ usecase_id: 1 }], error: null });
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag_id=2'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('empty keyword (q=) → no keyword filter applied, returns all', async () => {
    const chain = makeChain({ data: [UC_1, UC_2, UC_3], error: null, count: 3 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?q='));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.count).toBe(3);
  });

  test('search_by=title → applies title-only filter', async () => {
    const chain = makeChain({ data: [UC_1], error: null, count: 1 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?q=playground&search_by=title'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('search_by=description → applies description-only filter', async () => {
    const chain = makeChain({ data: [UC_2], error: null, count: 1 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?q=water&search_by=description'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('invalid page (negative) → 400 INVALID_PAGE', async () => {
    const res = await GET(makeRequest('http://localhost/api/usecases?page=-1'));
    const body = await res.json();
    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_PAGE');
  });

  test('invalid page (non-numeric) → 400 INVALID_PAGE', async () => {
    const res = await GET(makeRequest('http://localhost/api/usecases?page=abc'));
    const body = await res.json();
    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_PAGE');
  });

  test('invalid pageSize (over 100) → 400 INVALID_PAGE_SIZE', async () => {
    const res = await GET(makeRequest('http://localhost/api/usecases?pageSize=200'));
    const body = await res.json();
    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_PAGE_SIZE');
  });

  test('invalid pageSize (negative) → 400 INVALID_PAGE_SIZE', async () => {
    const res = await GET(makeRequest('http://localhost/api/usecases?pageSize=-5'));
    const body = await res.json();
    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_PAGE_SIZE');
  });

  test('invalid search_by value → 400 INVALID_SEARCH_BY', async () => {
    const res = await GET(makeRequest('http://localhost/api/usecases?search_by=invalid'));
    const body = await res.json();
    jestExpect(res.status).toBe(400);
    jestExpect(body.code).toBe('INVALID_SEARCH_BY');
  });


  test('tag_name not found → returns empty data with 200', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: [], error: null });
      return makeChain({ data: [], error: null, count: 0 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag_name=nonexistenttag'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.data).toHaveLength(0);
  });

});

// ─── Tests: Filter Combinations ──────────────────────────────────────────────

describe('GET /api/usecases - filter combinations', () => {

  test('keyword + category_id → returns filtered results', async () => {
    const chain = makeChain({ data: [UC_1], error: null, count: 1 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?q=playground&category_id=1'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.pagination.total).toBe(1);
  });

  test('category_id + tag_id → returns filtered results', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') return makeChain({ data: [{ usecase_id: 1 }], error: null });
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?category_id=1&tag_id=2'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('keyword + category_id + tag_ids → returns combined filtered results', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') {
        return makeChain({ data: [{ usecase_id: 1 }, { usecase_id: 2 }], error: null });
      }
      return makeChain({ data: [UC_1, UC_2], error: null, count: 2 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?q=park&category_id=1&tag_ids=2,4'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.pagination.total).toBe(2);
  });

  test('tag_id + tag_ids combined → deduplicates and filters correctly', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') {
        return makeChain({ data: [{ usecase_id: 1 }], error: null });
      }
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    // tag_id=2 and tag_ids=2,4 — 2 is duplicated, should deduplicate
    const res = await GET(makeRequest('http://localhost/api/usecases?tag_id=2&tag_ids=2,4'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('keyword + tag slug → returns filtered results', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: { id: 3 }, error: null });
      if (table === 'usecase_tags') return makeChain({ data: [{ usecase_id: 1 }], error: null });
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?q=playground&tag=family-friendly'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

  test('all filters combined → returns results matching all criteria', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'usecase_tags') {
        return makeChain({ data: [{ usecase_id: 1 }], error: null });
      }
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    const res = await GET(makeRequest(
      'http://localhost/api/usecases?q=playground&category_id=1&tag_ids=2,4&page=1&pageSize=5'
    ));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.pagination.pageSize).toBe(5);
  });

  test('no filters → returns all use cases with pagination', async () => {
    const chain = makeChain({ data: [UC_1, UC_2, UC_3], error: null, count: 3 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.count).toBe(3);
    jestExpect(body.pagination.total).toBe(3);
  });

  test('filter with no matching results → empty data with 200', async () => {
    const chain = makeChain({ data: [], error: null, count: 0 });
    (supabase.from as jest.Mock).mockReturnValue(chain);

    const res = await GET(makeRequest('http://localhost/api/usecases?q=nonexistentterm&category_id=99'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
    jestExpect(body.data).toHaveLength(0);
    jestExpect(body.pagination.total).toBe(0);
  });

  test('tag_name + category_id → returns filtered results', async () => {
    (supabase.from as jest.Mock).mockImplementation((table: string) => {
      if (table === 'tags') return makeChain({ data: [{ id: 2 }, { id: 4 }], error: null });
      if (table === 'usecase_tags') return makeChain({ data: [{ usecase_id: 1 }], error: null });
      return makeChain({ data: [UC_1], error: null, count: 1 });
    });

    const res = await GET(makeRequest('http://localhost/api/usecases?tag_name=park&category_id=1'));
    const body = await res.json();

    jestExpect(res.status).toBe(200);
    jestExpect(body.success).toBe(true);
  });

});
