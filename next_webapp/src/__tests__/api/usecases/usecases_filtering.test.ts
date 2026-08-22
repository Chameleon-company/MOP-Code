/**
 * @jest-environment node
 *
 * Tests for GET /api/usecases query construction.
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

jest.mock('../../../app/api/library/errorResponse', () => ({
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

const OBJECT_ID = '507f1f77bcf86cd799439011';

function makeRequest(url: string) {
  return { url } as unknown as Request;
}

function setupMongoResult(documents: unknown[] = [], total = 0) {
  const chain: any = {};

  chain.sort = jest.fn().mockReturnValue(chain);
  chain.skip = jest.fn().mockReturnValue(chain);
  chain.limit = jest.fn().mockReturnValue(chain);
  chain.lean = jest.fn().mockResolvedValue(documents);

  (UseCase.countDocuments as jest.Mock).mockResolvedValue(total);
  (UseCase.find as jest.Mock).mockReturnValue(chain);

  return chain;
}

function getMongoFilter() {
  return (UseCase.find as jest.Mock).mock.calls[0][0];
}

const UC_1 = {
  _id: '507f1f77bcf86cd799439021',
  title: 'Playground A',
  description: 'Kids park',
};

const UC_2 = {
  _id: '507f1f77bcf86cd799439022',
  title: 'Playground B',
  description: 'Water park',
};

const UC_3 = {
  _id: '507f1f77bcf86cd799439023',
  title: 'Tech Hub',
  description: 'Tech space',
};

beforeEach(() => {
  jest.clearAllMocks();
  (dbConnect as jest.Mock).mockResolvedValue(undefined);
  setupMongoResult();
});

describe('GET /api/usecases - MongoDB filters', () => {
  test('legacy category ID filters category.legacy_id', async () => {
    setupMongoResult([UC_1, UC_2], 2);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?category_id=1',
      ),
    );

    expect(response.status).toBe(200);
    expect(getMongoFilter()).toEqual({
      'category.legacy_id': '1',
    });
  });

  test('Mongo ObjectId category filters category.id', async () => {
    setupMongoResult([UC_1], 1);

    const response = await GET(
      makeRequest(
        `http://localhost/api/usecases?category_id=${OBJECT_ID}`,
      ),
    );

    expect(response.status).toBe(200);
    expect(getMongoFilter()).toEqual({
      'category.id': OBJECT_ID,
    });
  });

  test('non-ObjectId category remains a legacy ID', async () => {
    await GET(
      makeRequest(
        'http://localhost/api/usecases?category_id=abc',
      ),
    );

    expect(getMongoFilter()).toEqual({
      'category.legacy_id': 'abc',
    });
  });

  test('legacy tag ID filters tags.legacy_id', async () => {
    setupMongoResult([UC_1], 1);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?tag_id=2',
      ),
    );

    expect(response.status).toBe(200);
    expect(getMongoFilter()).toEqual({
      $or: [
        {
          'tags.legacy_id': {
            $in: ['2'],
          },
        },
      ],
    });
  });

  test('supports Mongo and legacy tag IDs together', async () => {
    setupMongoResult([UC_1], 1);

    const response = await GET(
      makeRequest(
        `http://localhost/api/usecases?tag_ids=${OBJECT_ID},2,abc`,
      ),
    );

    expect(response.status).toBe(200);
    expect(getMongoFilter()).toEqual({
      $or: [
        {
          'tags.id': {
            $in: [OBJECT_ID],
          },
        },
        {
          'tags.legacy_id': {
            $in: ['2', 'abc'],
          },
        },
      ],
    });
  });

  test('tag slug filters the embedded tags.slug field', async () => {
    await GET(
      makeRequest(
        'http://localhost/api/usecases?tag=family-friendly',
      ),
    );

    expect(getMongoFilter()).toEqual({
      'tags.slug': 'family-friendly',
    });
  });

  test('tag name creates a case-insensitive regular expression', async () => {
    setupMongoResult([], 0);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?tag_name=park',
      ),
    );
    const body = await response.json();
    const filter = getMongoFilter();

    expect(response.status).toBe(200);
    expect(body.data).toEqual([]);
    expect(filter['tags.name']).toEqual(expect.any(RegExp));
    expect(filter['tags.name'].source).toBe('park');
    expect(filter['tags.name'].flags).toContain('i');
  });

  test('empty keyword does not add a search filter', async () => {
    setupMongoResult([UC_1, UC_2, UC_3], 3);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?q=',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.count).toBe(3);
    expect(getMongoFilter()).toEqual({});
  });

  test('search_by title filters title only', async () => {
    setupMongoResult([UC_1], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?q=playground&search_by=title',
      ),
    );

    const filter = getMongoFilter();

    expect(filter.title).toEqual(expect.any(RegExp));
    expect(filter.title.source).toBe('playground');
    expect(filter.title.flags).toContain('i');
    expect(filter.description).toBeUndefined();
    expect(filter.$or).toBeUndefined();
  });

  test('search_by description filters description only', async () => {
    setupMongoResult([UC_2], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?q=water&search_by=description',
      ),
    );

    const filter = getMongoFilter();

    expect(filter.description).toEqual(expect.any(RegExp));
    expect(filter.description.source).toBe('water');
    expect(filter.description.flags).toContain('i');
    expect(filter.title).toBeUndefined();
    expect(filter.$or).toBeUndefined();
  });

  test('default keyword search covers title and description', async () => {
    setupMongoResult([UC_1], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?q=playground',
      ),
    );

    const filter = getMongoFilter();

    expect(filter.$or).toHaveLength(2);
    expect(filter.$or[0].title).toEqual(expect.any(RegExp));
    expect(filter.$or[1].description).toEqual(expect.any(RegExp));
  });

  test('escapes regular expression characters in keyword input', async () => {
    await GET(
      makeRequest(
        'http://localhost/api/usecases?q=park%2Btest',
      ),
    );

    const filter = getMongoFilter();

    expect(filter.$or[0].title.source).toBe('park\\+test');
    expect(filter.$or[1].description.source).toBe(
      'park\\+test',
    );
  });
});

describe('GET /api/usecases - validation', () => {
  test('negative page returns 400 INVALID_PAGE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?page=-1',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE');
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('non-numeric page returns 400 INVALID_PAGE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?page=abc',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE');
    expect(UseCase.find).not.toHaveBeenCalled();
  });

  test('pageSize over 100 returns 400 INVALID_PAGE_SIZE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=200',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE_SIZE');
  });

  test('negative pageSize returns 400 INVALID_PAGE_SIZE', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?pageSize=-5',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_PAGE_SIZE');
  });

  test('invalid search_by returns 400 INVALID_SEARCH_BY', async () => {
    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?search_by=invalid',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_SEARCH_BY');
    expect(dbConnect).not.toHaveBeenCalled();
  });
});

describe('GET /api/usecases - combined filters', () => {
  test('combines keyword and legacy category ID', async () => {
    setupMongoResult([UC_1], 1);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?q=playground&category_id=1',
      ),
    );
    const body = await response.json();
    const filter = getMongoFilter();

    expect(response.status).toBe(200);
    expect(body.pagination.total).toBe(1);
    expect(filter['category.legacy_id']).toBe('1');
    expect(filter.$or).toHaveLength(2);
  });

  test('combines category and tag ID filters', async () => {
    setupMongoResult([UC_1], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?category_id=1&tag_id=2',
      ),
    );

    expect(getMongoFilter()).toEqual({
      'category.legacy_id': '1',
      $or: [
        {
          'tags.legacy_id': {
            $in: ['2'],
          },
        },
      ],
    });
  });

  test('combines keyword and tag IDs using $and', async () => {
    setupMongoResult([UC_1, UC_2], 2);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?q=park&tag_ids=2,4',
      ),
    );
    const body = await response.json();
    const filter = getMongoFilter();

    expect(response.status).toBe(200);
    expect(body.pagination.total).toBe(2);
    expect(filter.$or).toBeUndefined();
    expect(filter.$and).toHaveLength(2);

    expect(filter.$and[0].$or[0].title).toEqual(
      expect.any(RegExp),
    );
    expect(filter.$and[0].$or[1].description).toEqual(
      expect.any(RegExp),
    );
    expect(filter.$and[1]).toEqual({
      $or: [
        {
          'tags.legacy_id': {
            $in: ['2', '4'],
          },
        },
      ],
    });
  });

  test('combines keyword and tag slug', async () => {
    setupMongoResult([UC_1], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?q=playground&tag=family-friendly',
      ),
    );

    const filter = getMongoFilter();

    expect(filter['tags.slug']).toBe('family-friendly');
    expect(filter.$or).toHaveLength(2);
  });

  test('combines all filters and pagination', async () => {
    const chain = setupMongoResult([UC_1], 1);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?q=playground&category_id=1&tag_ids=2,4&page=1&pageSize=5',
      ),
    );
    const body = await response.json();
    const filter = getMongoFilter();

    expect(response.status).toBe(200);
    expect(body.pagination.pageSize).toBe(5);
    expect(filter['category.legacy_id']).toBe('1');
    expect(filter.$and).toHaveLength(2);
    expect(chain.skip).toHaveBeenCalledWith(0);
    expect(chain.limit).toHaveBeenCalledWith(5);
  });

  test('no filters queries MongoDB with an empty filter', async () => {
    setupMongoResult([UC_1, UC_2, UC_3], 3);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.count).toBe(3);
    expect(body.pagination.total).toBe(3);
    expect(getMongoFilter()).toEqual({});
  });

  test('no matching documents returns an empty successful response', async () => {
    setupMongoResult([], 0);

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases?q=missing&category_id=99',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual([]);
    expect(body.pagination.total).toBe(0);
  });

  test('combines tag name and category ID', async () => {
    setupMongoResult([UC_1], 1);

    await GET(
      makeRequest(
        'http://localhost/api/usecases?tag_name=park&category_id=1',
      ),
    );

    const filter = getMongoFilter();

    expect(filter['category.legacy_id']).toBe('1');
    expect(filter['tags.name']).toEqual(expect.any(RegExp));
    expect(filter['tags.name'].source).toBe('park');
  });

  test('database failure returns 500 INTERNAL_ERROR', async () => {
    (dbConnect as jest.Mock).mockRejectedValue(
      new Error('Database unavailable'),
    );

    const response = await GET(
      makeRequest(
        'http://localhost/api/usecases',
      ),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.code).toBe('INTERNAL_ERROR');
  });
});