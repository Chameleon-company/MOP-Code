/**
 * @jest-environment node
 *
 * Tests for GET /api/usecases pagination and tag filtering.
 * MongoDB operations are mocked, so no real database work occurs.
 */

jest.mock('next/server', () => ({
  NextResponse: {
    json: jest.fn().mockImplementation(
      (body: unknown, init?: { status?: number }) => ({
        status: init?.status ?? 200,
        json: jest.fn().mockResolvedValue(body),
        _body: body,
      }),
    ),
  },
}));

jest.mock('@/lib/dbConnect', () => ({
  __esModule: true,
  default: jest.fn(),
}));

jest.mock('@/models/mongoose/UseCase', () => ({
  __esModule: true,
  default: {
    countDocuments: jest.fn(),
    find: jest.fn(),
  },
}));

jest.mock('@/app/api/library/useCaseDto', () => ({
  toUseCaseDTO: jest.fn((document: unknown) => document),
}));

jest.mock('@/app/api/library/errorResponse', () => ({
  errorResponse: jest.fn().mockImplementation(
    (message: string, status: number, code?: string) => ({
      status,
      json: jest.fn().mockResolvedValue({
        success: false,
        message,
        code,
      }),
      _body: {
        success: false,
        message,
        code,
      },
    }),
  ),
}));

import { GET } from '../../../app/api/usecases/route';
import dbConnect from '@/lib/dbConnect';
import UseCase from '@/models/mongoose/UseCase';

function makeRequest(url: string) {
  return { url } as unknown as Request;
}

function setupMongoResult(documents: unknown[], total: number) {
  const chain: any = {};

  chain.sort = jest.fn().mockReturnValue(chain);
  chain.skip = jest.fn().mockReturnValue(chain);
  chain.limit = jest.fn().mockReturnValue(chain);
  chain.lean = jest.fn().mockResolvedValue(documents);

  (UseCase.countDocuments as jest.Mock).mockResolvedValue(total);
  (UseCase.find as jest.Mock).mockReturnValue(chain);

  return chain;
}

const UC_1 = {
  _id: '507f1f77bcf86cd799439011',
  title: 'ML Basics',
  description: 'Intro to ML',
  tags: [{ name: 'Machine Learning', slug: 'ml' }],
};

const UC_2 = {
  _id: '507f1f77bcf86cd799439012',
  title: 'Deep Learning',
  description: 'Neural networks',
  tags: [{ name: 'Machine Learning', slug: 'ml' }],
};

const UC_3 = {
  _id: '507f1f77bcf86cd799439013',
  title: 'Climate',
  description: 'Weather models',
  tags: [{ name: 'Environment', slug: 'environment' }],
};

beforeEach(() => {
  jest.clearAllMocks();
  (dbConnect as jest.Mock).mockResolvedValue(undefined);
  setupMongoResult([], 0);
});

describe('GET /api/usecases - pagination', () => {
  test('uses page 1 and pageSize 10 by default', async () => {
    const chain = setupMongoResult(
      [UC_1, UC_2, UC_3],
      3,
    );

    const response = await GET(
      makeRequest('http://localhost/api/usecases'),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.count).toBe(3);
    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 10,
      total: 3,
      totalPages: 1,
    });

    expect(dbConnect).toHaveBeenCalledTimes(1);
    expect(chain.sort).toHaveBeenCalledWith({
      created_at: -1,
    });
    expect(chain.skip).toHaveBeenCalledWith(0);
    expect(chain.limit).toHaveBeenCalledWith(10);
  });

  test('applies page 2 and pageSize 5', async () => {
    const chain = setupMongoResult([UC_3], 11);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?page=2&pageSize=5',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.pagination).toEqual({
      page: 2,
      pageSize: 5,
      total: 11,
      totalPages: 3,
    });

    expect(chain.skip).toHaveBeenCalledWith(5);
    expect(chain.limit).toHaveBeenCalledWith(5);
  });

  test('rounds totalPages up', async () => {
    setupMongoResult([UC_1, UC_2, UC_3], 7);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=3',
      ),
    );
    const body = await response.json();

    expect(body.pagination.totalPages).toBe(3);
  });

  test('invalid page returns 400 INVALID_PAGE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?page=abc',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE');
    expect(dbConnect).not.toHaveBeenCalled();
    expect(UseCase.find).not.toHaveBeenCalled();
  });

  test('invalid pageSize returns 400 INVALID_PAGE_SIZE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=0',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE_SIZE');
    expect(dbConnect).not.toHaveBeenCalled();
    expect(UseCase.find).not.toHaveBeenCalled();
  });

  test('pageSize above 100 returns 400', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=101',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE_SIZE');
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('returns zero pagination values when no results exist', async () => {
    setupMongoResult([], 0);

    const response = await GET(
      makeRequest('http://localhost/api/usecases'),
    );
    const body = await response.json();

    expect(body.success).toBe(true);
    expect(body.data).toEqual([]);
    expect(body.count).toBe(0);
    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 10,
      total: 0,
      totalPages: 0,
    });
  });

  test('count equals the number of documents on the current page', async () => {
    setupMongoResult([UC_1, UC_2], 50);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=2',
      ),
    );
    const body = await response.json();

    expect(body.count).toBe(2);
    expect(body.pagination.total).toBe(50);
    expect(body.pagination.totalPages).toBe(25);
  });
});

describe('GET /api/usecases - embedded tag filtering', () => {
  test('filters directly by embedded tag slug', async () => {
    setupMongoResult([UC_1, UC_2], 2);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?tag=ml&pageSize=5',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual([UC_1, UC_2]);
    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 5,
      total: 2,
      totalPages: 1,
    });

    expect(UseCase.countDocuments).toHaveBeenCalledWith({
      'tags.slug': 'ml',
    });
    expect(UseCase.find).toHaveBeenCalledWith({
      'tags.slug': 'ml',
    });
  });

  test('returns an empty result when a tag has no matching use cases', async () => {
    setupMongoResult([], 0);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?tag=ghost',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual([]);
    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 10,
      total: 0,
      totalPages: 0,
    });

    expect(UseCase.find).toHaveBeenCalledWith({
      'tags.slug': 'ghost',
    });
  });

  test('applies pagination together with the tag filter', async () => {
    const chain = setupMongoResult([], 12);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?tag=ml&page=3&pageSize=5',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.pagination).toEqual({
      page: 3,
      pageSize: 5,
      total: 12,
      totalPages: 3,
    });

    expect(chain.skip).toHaveBeenCalledWith(10);
    expect(chain.limit).toHaveBeenCalledWith(5);
  });
});