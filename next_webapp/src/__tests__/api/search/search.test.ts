/**
 * @jest-environment node
 *
 * Tests for GET /api/search (MongoDB / Mongoose version).
 *
 * All external dependencies are mocked — no real DB calls happen.
 * Key regression tested here: content / content_file_id must NEVER appear
 * in search results (the old Supabase route exposed these via select('*')).
 */

// ── Next.js mock ──────────────────────────────────────────────────────────────
jest.mock('next/server', () => ({
  NextResponse: {
    json: jest.fn().mockImplementation((body: unknown, init?: { status?: number }) => ({
      status: init?.status ?? 200,
      json: jest.fn().mockResolvedValue(body),
      _body: body,
    })),
  },
}));

// ── dbConnect mock ────────────────────────────────────────────────────────────
jest.mock('@/lib/dbConnect', () => ({
  dbConnect: jest.fn().mockResolvedValue(undefined),
}));

// ── errorResponse mock ────────────────────────────────────────────────────────
jest.mock('@/app/api/library/errorResponse', () => ({
  errorResponse: jest.fn().mockImplementation(
    (message: string, status: number, code?: string) => ({
      status,
      json: jest.fn().mockResolvedValue({ success: false, message, code }),
      _body: { success: false, message, code },
    }),
  ),
}));

// ── UseCase model mock ────────────────────────────────────────────────────────
// The route counts matching documents, then calls
// UseCase.find(filter).select(...).sort(...).skip(...).limit(...).lean().
// The mutable values let individual tests inject a page of results and its
// independently counted total.
let __mockResults: unknown[] = [];
let __mockTotal = 0;

jest.mock('@/models/mongoose/UseCase', () => {
  const chain = {
    select: jest.fn().mockReturnThis(),
    sort:   jest.fn().mockReturnThis(),
    skip:   jest.fn().mockReturnThis(),
    limit:  jest.fn().mockReturnThis(),
    lean:   jest.fn().mockImplementation(() => Promise.resolve(__mockResults)),
  };
  return {
    UseCase: {
      countDocuments: jest.fn().mockImplementation(() => Promise.resolve(__mockTotal)),
      find: jest.fn().mockReturnValue(chain),
      aggregate: jest.fn().mockImplementation(() => ({
        exec: jest.fn().mockImplementation(() => Promise.resolve(__mockResults)),
      })),
    },
  };
});

// ── mongoose mock (for ObjectId / FilterQuery) ────────────────────────────────
jest.mock('mongoose', () => {
  const actual = jest.requireActual('mongoose') as typeof import('mongoose');
  return {
    ...actual,
    FilterQuery: actual.FilterQuery,
    Types: {
      ...actual.Types,
      ObjectId: class FakeObjectId {
        private val: string;
        constructor(v: string) { this.val = v; }
        toString() { return this.val; }
      },
    },
  };
});

// ── Imports (after mocks) ─────────────────────────────────────────────────────
import { GET } from '../../../app/api/search/route';
import { UseCase } from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';

// ── Helpers ───────────────────────────────────────────────────────────────────
function makeRequest(url: string) {
  return { url } as unknown as import('next/server').NextRequest;
}

function setResults(docs: unknown[], total = docs.length) {
  __mockResults = docs;
  __mockTotal = total;
}

// ── Fixtures ──────────────────────────────────────────────────────────────────
// These documents deliberately include NO content / content_file_id, mirroring
// what the MongoDB projection should return.
const UC_1 = {
  _id: 'id1', legacy_id: '1', title: 'ML Basics',
  description: 'Intro to machine learning',
  category: { legacy_id: '1', category_name: 'AI' },
  tags: [{ slug: 'ml', name: 'Machine Learning' }],
  created_at: '2024-01-01', updated_at: '2024-01-01',
};
const UC_2 = {
  _id: 'id2', legacy_id: '2', title: 'Deep Learning',
  description: 'Neural network fundamentals',
  category: { legacy_id: '3', category_name: 'Data Science' },
  tags: [{ slug: 'dl', name: 'Deep Learning' }, { slug: 'ml', name: 'Machine Learning' }],
  created_at: '2024-01-02', updated_at: '2024-01-02',
};
const UC_3 = {
  _id: 'id3', legacy_id: '3', title: 'Climate Forecast',
  description: 'Weather prediction models',
  category: { legacy_id: '2', category_name: 'Environment' },
  tags: [{ slug: 'climate', name: 'Climate' }],
  created_at: '2024-01-03', updated_at: '2024-01-03',
};

// ── Tests ─────────────────────────────────────────────────────────────────────
beforeEach(() => {
  jest.clearAllMocks();
  __mockResults = [];
  __mockTotal = 0;
});

describe('GET /api/search', () => {

  // ── Basic listing ───────────────────────────────────────────────────────────
  test('no params — returns all use cases', async () => {
    setResults([UC_1, UC_2, UC_3]);
    const res = await GET(makeRequest('http://localhost/api/search'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data.results).toHaveLength(3);
    expect(body.data.pagination.total).toBe(3);
    expect(body.data.filters).toEqual({ q: null, title: null, category: null, tag: null });
  });

  // ── Keyword search (q) ──────────────────────────────────────────────────────
  test('q — passes a $or regex filter to UseCase.find', async () => {
    setResults([UC_1, UC_2]);
    const res = await GET(makeRequest('http://localhost/api/search?q=ml'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(2);
    // Confirm the filter passed to .find() contains $or
    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    expect(findArg).toHaveProperty('$or');
    expect(Array.isArray(findArg.$or)).toBe(true);
    expect(UseCase.countDocuments).toHaveBeenCalledTimes(1);
    expect(UseCase.find).toHaveBeenCalledTimes(1);
  });

  // ── Title-only search ───────────────────────────────────────────────────────
  test('title — passes a title regex filter (no $or)', async () => {
    setResults([UC_2]);
    const res = await GET(makeRequest('http://localhost/api/search?title=deep'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    expect(findArg).toHaveProperty('title');
    expect(findArg).not.toHaveProperty('$or');
    expect(UseCase.find).toHaveBeenCalledTimes(1);
  });

  // ── q takes priority over title ─────────────────────────────────────────────
  test('q takes priority over title when both are present', async () => {
    setResults([UC_1]);
    await GET(makeRequest('http://localhost/api/search?q=ml&title=deep'));

    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    // q is present → must use $or, not a bare title filter
    expect(findArg).toHaveProperty('$or');
    expect(findArg).not.toHaveProperty('title');
  });

  // ── Category filter (legacy_id) ─────────────────────────────────────────────
  test('category (numeric) — filters by category.legacy_id', async () => {
    setResults([UC_2]);
    const res = await GET(makeRequest('http://localhost/api/search?category=3'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    expect(findArg['category.legacy_id']).toBe('3');
  });

  // ── Category filter (ObjectId) ──────────────────────────────────────────────
  test('category (24-hex ObjectId) — filters by category.id', async () => {
    setResults([UC_1]);
    const oid = 'a'.repeat(24);
    await GET(makeRequest(`http://localhost/api/search?category=${oid}`));

    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    // The key is literally 'category.id' (a dot in the key name, not a nested path).
    // Use bracket access instead of toHaveProperty so Jest doesn't traverse the dot.
    expect(findArg['category.id']).toBeDefined();
    expect(findArg['category.legacy_id']).toBeUndefined();
  });

  // ── Invalid category ────────────────────────────────────────────────────────
  test('invalid category — returns 400 INVALID_CATEGORY without querying DB', async () => {
    await GET(makeRequest('http://localhost/api/search?category=notanumber'));

    expect(errorResponse).toHaveBeenCalledWith(
      'category must be a valid integer or MongoDB ObjectId',
      400,
      'INVALID_CATEGORY',
    );
    expect(UseCase.countDocuments).not.toHaveBeenCalled();
    expect(UseCase.find).not.toHaveBeenCalled();
  });

  // ── Tag filter ──────────────────────────────────────────────────────────────
  test('tag — uses the embedded tags.slug filter for count and results', async () => {
    setResults([UC_1, UC_2]);
    const res = await GET(makeRequest('http://localhost/api/search?tag=ml'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(2);
    expect(UseCase.countDocuments).toHaveBeenCalledTimes(1);
    expect(UseCase.find).toHaveBeenCalledTimes(1);
    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    expect(findArg['tags.slug']).toBe('ml');
    expect(UseCase.countDocuments).toHaveBeenCalledWith(findArg);
  });

  // ── Unknown tag slug ────────────────────────────────────────────────────────
  test('unknown tag slug — returns empty results gracefully', async () => {
    setResults([]);
    const res = await GET(makeRequest('http://localhost/api/search?tag=ghost-tag'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(0);
    expect(body.data.pagination.total).toBe(0);
  });

  // ── Combined filters ────────────────────────────────────────────────────────
  test('q + category + tag — the same combined filter is used for count and results', async () => {
    setResults([UC_2]);
    const res = await GET(makeRequest('http://localhost/api/search?q=deep&category=3&tag=dl'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    expect(UseCase.find).toHaveBeenCalledTimes(1);
    const findArg = (UseCase.find as jest.Mock).mock.calls[0][0];
    expect(findArg).toHaveProperty('$or');
    expect(findArg['category.legacy_id']).toBe('3');
    expect(findArg['tags.slug']).toBe('dl');
    expect(UseCase.countDocuments).toHaveBeenCalledWith(findArg);
  });

  // ── Pagination ──────────────────────────────────────────────────────────────
  test('pagination — fetches only the requested page from MongoDB', async () => {
    setResults([UC_2], 3);
    const res = await GET(makeRequest('http://localhost/api/search?page=2&pageSize=1'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toHaveLength(1);
    expect(body.data.results[0]._id).toBe('id2');
    expect(body.data.pagination).toMatchObject({
      page: 2, pageSize: 1, total: 3, totalPages: 3,
      hasNext: true, hasPrev: true,
    });

    const chainInstance = (UseCase.find as jest.Mock).mock.results[0].value;
    expect(chainInstance.skip).toHaveBeenCalledWith(1);
    expect(chainInstance.limit).toHaveBeenCalledWith(1);
  });

  test('pagination — clamps an out-of-range page before querying MongoDB', async () => {
    setResults([UC_3], 3);
    const res = await GET(makeRequest('http://localhost/api/search?page=99&pageSize=2'));
    const body = await res.json();

    expect(body.data.pagination).toMatchObject({
      page: 2, pageSize: 2, total: 3, totalPages: 2,
      hasNext: false, hasPrev: true,
    });

    const chainInstance = (UseCase.find as jest.Mock).mock.results[0].value;
    expect(chainInstance.skip).toHaveBeenCalledWith(2);
    expect(chainInstance.limit).toHaveBeenCalledWith(2);
  });

  // ── Content leak regression ─────────────────────────────────────────────────
  test('SECURITY: .select() projection excludes content and content_file_id', async () => {
    // The route must pass a projection to .select() that does NOT include
    // `content` or `content_file_id`.  Mongoose enforces the projection
    // server-side; this test verifies the route passes the right whitelist.
    setResults([UC_1]);

    await GET(makeRequest('http://localhost/api/search'));

    // Retrieve the chain returned by UseCase.find() and inspect .select() call.
    const chainInstance = (UseCase.find as jest.Mock).mock.results[0].value;
    expect(chainInstance.select).toHaveBeenCalledTimes(1);

    const projection = chainInstance.select.mock.calls[0][0] as Record<string, number>;

    // Whitelisted safe fields must be present with value 1.
    expect(projection._id).toBe(1);
    expect(projection.title).toBe(1);
    expect(projection.description).toBe(1);
    expect(projection.category).toBe(1);
    expect(projection.tags).toBe(1);

    // Forbidden fields must be absent from the projection entirely.
    expect(projection).not.toHaveProperty('content');
    expect(projection).not.toHaveProperty('content_file_id');
  });

  test('sort memory limit — retries with safe projection before sorting the last page', async () => {
    setResults([UC_3], 3);
    const chain = (UseCase.find as jest.Mock)();
    chain.lean.mockRejectedValueOnce(Object.assign(new Error('Sort exceeded memory limit'), { code: 292 }));

    const res = await GET(makeRequest('http://localhost/api/search?page=999&pageSize=2&tag=ml&sortBy=title&sortOrder=ASC'));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.data.results).toEqual([UC_3]);
    expect(body.data.pagination).toMatchObject({ page: 2, total: 3, hasNext: false });
    const pipeline = (UseCase.aggregate as jest.Mock).mock.calls[0][0];
    expect(pipeline).toEqual([
      { $match: { 'tags.slug': 'ml' } },
      { $project: expect.objectContaining({ _id: 1, title: 1 }) },
      { $sort: { title: 1 } },
      { $skip: 2 },
      { $limit: 2 },
    ]);
    expect(pipeline[1].$project).not.toHaveProperty('content');
    expect(pipeline[1].$project).not.toHaveProperty('content_file_id');
  });

  test('sort fallback failure — returns 500 INTERNAL_ERROR', async () => {
    const chain = (UseCase.find as jest.Mock)();
    chain.lean.mockRejectedValueOnce(Object.assign(new Error('Sort exceeded memory limit'), { code: 292 }));
    (UseCase.aggregate as jest.Mock).mockImplementationOnce(() => ({
      exec: jest.fn().mockRejectedValue(new Error('Aggregation failed')),
    }));

    const res = await GET(makeRequest('http://localhost/api/search'));
    expect(res.status).toBe(500);
  });

  // ── Internal error ──────────────────────────────────────────────────────────
  test('DB error — returns 500 INTERNAL_ERROR', async () => {
    (UseCase.find as jest.Mock).mockImplementation(() => {
      throw new Error('Connection refused');
    });

    await GET(makeRequest('http://localhost/api/search'));

    expect(errorResponse).toHaveBeenCalledWith('Internal server error', 500, 'INTERNAL_ERROR');
  });

});
