


/**
 * @jest-environment node
 */
 
import { testApiHandler } from 'next-test-api-route-handler';
import * as handler from '@/app/api/usecases/[id]/tags/route';
 
// ─── Supabase mock ────────────────────────────────────────────────────────────
const mockEq = jest.fn();
const mockIn = jest.fn();
 
jest.mock('@/library/supabaseClient', () => ({
  supabase: {
    from: jest.fn((table: string) => {
      if (table === 'usecase_tags') {
        return { select: jest.fn().mockReturnValue({ eq: mockEq }) };
      }
      if (table === 'tags') {
        return { select: jest.fn().mockReturnValue({ in: mockIn }) };
      }
    }),
  },
}));
 
// ─── errorResponse mock ───────────────────────────────────────────────────────
jest.mock('@/app/api/library/errorResponse', () => ({
  errorResponse: (message: string, status: number, code: string) =>
    new Response(JSON.stringify({ error: message, code }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
}));
 
// ─── Helper ───────────────────────────────────────────────────────────────────
async function callRoute(id: string) {
  let status = 0;
  let body: Record<string, unknown> = {};
 
  await testApiHandler({
    appHandler: handler,
    params: { id },
    async test({ fetch }) {
      const res = await fetch({ method: 'GET' });
      status = res.status;
      body = await res.json();
    },
  });
 
  return { status, body };
}
 
// ─── Tests ────────────────────────────────────────────────────────────────────
beforeEach(() => jest.clearAllMocks());
 
// ── 1. ID Validation ──────────────────────────────────────────────────────────
describe('ID validation', () => {
  const invalidIds: [string, string][] = [
    ['non-numeric string', 'abc'],
    ['zero',               '0'],
    ['negative number',    '-3'],
    ['float',              '2.7'],
    ['empty string',       ''],
  ];
 
  test.each(invalidIds)('%s → 400 INVALID_ID', async (_label, id) => {
    const { status, body } = await callRoute(id);
    expect(status).toBe(400);
    expect(body.code).toBe('INVALID_ID');
    expect(mockEq).not.toHaveBeenCalled();
  });
 
  test('valid positive integer passes validation and hits DB', async () => {
    mockEq.mockResolvedValueOnce({ data: [], error: null });
    const { status } = await callRoute('5');
    // 404 means validation passed and DB was reached
    expect(status).toBe(404);
    expect(mockEq).toHaveBeenCalledTimes(1);
  });
});
 
// ── 2. usecase_tags DB error ──────────────────────────────────────────────────
describe('usecase_tags query failure', () => {
  test('returns 500 INTERNAL_ERROR when DB throws', async () => {
    mockEq.mockResolvedValueOnce({
      data: null,
      error: { message: 'connection refused', code: 'PGRST000' },
    });
 
    const { status, body } = await callRoute('1');
    expect(status).toBe(500);
    expect(body.code).toBe('INTERNAL_ERROR');
    expect(mockIn).not.toHaveBeenCalled();
  });
});
 
// ── 3. No tags linked to use case ─────────────────────────────────────────────
describe('empty usecase_tags result', () => {
  test('returns 404 NOT_FOUND when data is empty array', async () => {
    mockEq.mockResolvedValueOnce({ data: [], error: null });
    const { status, body } = await callRoute('42');
    expect(status).toBe(404);
    expect(body.code).toBe('NOT_FOUND');
    expect(mockIn).not.toHaveBeenCalled();
  });
 
  test('returns 404 NOT_FOUND when data is null', async () => {
    mockEq.mockResolvedValueOnce({ data: null, error: null });
    const { status, body } = await callRoute('42');
    expect(status).toBe(404);
    expect(body.code).toBe('NOT_FOUND');
  });
});
 
// ── 4. tags table DB error ────────────────────────────────────────────────────
describe('tags fetch failure', () => {
  test('returns 500 INTERNAL_ERROR when tags table query fails', async () => {
    mockEq.mockResolvedValueOnce({
      data: [{ tag_id: 10 }, { tag_id: 11 }],
      error: null,
    });
    mockIn.mockResolvedValueOnce({
      data: null,
      error: { message: 'tags table unavailable', code: 'PGRST000' },
    });
 
    const { status, body } = await callRoute('1');
    expect(status).toBe(500);
    expect(body.code).toBe('INTERNAL_ERROR');
  });
});
 
// ── 5. Nullish coalescing fallback ────────────────────────────────────────────
describe('nullish coalescing on tags data', () => {
  test('returns [] when tags query returns null with no error', async () => {
    mockEq.mockResolvedValueOnce({ data: [{ tag_id: 10 }], error: null });
    mockIn.mockResolvedValueOnce({ data: null, error: null });
 
    const { status, body } = await callRoute('1');
    expect(status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual([]);
  });
});
 
// ── 6. Happy path ─────────────────────────────────────────────────────────────
describe('successful tag retrieval', () => {
  test('returns correct tags array with all fields', async () => {
    mockEq.mockResolvedValueOnce({
      data: [{ tag_id: 10 }, { tag_id: 11 }],
      error: null,
    });
    mockIn.mockResolvedValueOnce({
      data: [
        { id: 10, name: 'machine learning', slug: 'machine-learning' },
        { id: 11, name: 'healthcare',       slug: 'healthcare' },
      ],
      error: null,
    });
 
    const { status, body } = await callRoute('1');
    expect(status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toHaveLength(2);
    expect(body.data[0]).toEqual({ id: 10, name: 'machine learning', slug: 'machine-learning' });
    expect(body.data[1]).toEqual({ id: 11, name: 'healthcare', slug: 'healthcare' });
  });
 
  test('passes correct tag IDs into the .in() call', async () => {
    mockEq.mockResolvedValueOnce({
      data: [{ tag_id: 7 }, { tag_id: 8 }, { tag_id: 9 }],
      error: null,
    });
    mockIn.mockResolvedValueOnce({
      data: [
        { id: 7, name: 'fintech',    slug: 'fintech' },
        { id: 8, name: 'ai',         slug: 'ai' },
        { id: 9, name: 'legal',      slug: 'legal' },
      ],
      error: null,
    });
 
    await callRoute('3');
    expect(mockIn).toHaveBeenCalledWith('id', [7, 8, 9]);
  });
 
  test('single tag returns correctly', async () => {
    mockEq.mockResolvedValueOnce({ data: [{ tag_id: 5 }], error: null });
    mockIn.mockResolvedValueOnce({
      data: [{ id: 5, name: 'blockchain', slug: 'blockchain' }],
      error: null,
    });
 
    const { status, body } = await callRoute('99');
    expect(status).toBe(200);
    expect(body.data).toHaveLength(1);
    expect(body.data[0].slug).toBe('blockchain');
  });
});