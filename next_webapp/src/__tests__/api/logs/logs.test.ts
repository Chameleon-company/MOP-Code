/**
 * @jest-environment node
 *
 * Tests for the MongoDB-backed /api/logs routes.
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

jest.mock('@/models/mongoose/Log', () => ({
  __esModule: true,
  default: {
    find: jest.fn(),
    countDocuments: jest.fn(),
    deleteMany: jest.fn(),
  },
}));

jest.mock('@/models/mongoose/User', () => ({
  __esModule: true,
  default: {
    findOne: jest.fn(),
  },
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

import { Types } from 'mongoose';
import { GET, DELETE } from '../../../app/api/logs/route';
import dbConnect from '@/lib/dbConnect';
import Log from '@/models/mongoose/Log';
import User from '@/models/mongoose/User';
import { getAuthUser } from '@/app/api/library/auth';

function makeRequest(url = 'http://localhost:3000/api/logs') {
  return {
    url,
    headers: {
      get: () => null,
    },
  } as any;
}

function makeFindChain(result: unknown[]) {
  const chain: Record<string, jest.Mock> = {};

  chain.sort = jest.fn().mockReturnValue(chain);
  chain.skip = jest.fn().mockReturnValue(chain);
  chain.limit = jest.fn().mockReturnValue(chain);
  chain.populate = jest.fn().mockReturnValue(chain);
  chain.lean = jest.fn().mockResolvedValue(result);

  return chain;
}

const ADMIN_AUTH = {
  userId: 9,
  roleId: 1,
  roleName: 'admin',
  isAuthenticated: true,
  isAdmin: true,
};

const USER_AUTH = {
  userId: 7,
  roleId: 2,
  roleName: 'user',
  isAuthenticated: true,
  isAdmin: false,
};

const ANON_AUTH = {
  userId: null,
  roleId: null,
  roleName: null,
  isAuthenticated: false,
  isAdmin: false,
};

const MOCK_LOGS = [
  {
    _id: 'mongo-log-1',
    legacy_id: '1',
    level: 'info',
    message: 'Request logged',
    source: 'middleware',
    timestamp: new Date('2026-01-01T00:00:00.000Z'),
    user_id: {
      _id: 'mongo-user-42',
      legacy_id: '42',
    },
  },
  {
    _id: 'mongo-log-2',
    legacy_id: '2',
    level: 'error',
    message: 'DB error',
    source: 'api',
    timestamp: new Date('2026-01-01T00:01:00.000Z'),
    user_id: null,
  },
];

describe('GET /api/logs', () => {
  let findChain: Record<string, jest.Mock>;
  let userFindOneLean: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();

    (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
    (dbConnect as jest.Mock).mockResolvedValue(undefined);

    findChain = makeFindChain(MOCK_LOGS);
    (Log.find as jest.Mock).mockReturnValue(findChain);
    (Log.countDocuments as jest.Mock).mockResolvedValue(2);

    userFindOneLean = jest.fn().mockResolvedValue(null);

    (User.findOne as jest.Mock).mockReturnValue({
      select: jest.fn().mockReturnValue({
        lean: userFindOneLean,
      }),
    });
  });

  test('returns 401 for an unauthenticated request', async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.success).toBe(false);
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('returns 403 for an authenticated non-admin user', async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(403);
    expect(body.success).toBe(false);
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('returns logs with the existing API and pagination shape', async () => {
    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);

    expect(body.data).toEqual([
      {
        id: 1,
        level: 'info',
        message: 'Request logged',
        source: 'middleware',
        timestamp: new Date('2026-01-01T00:00:00.000Z'),
        user_id: 42,
      },
      {
        id: 2,
        level: 'error',
        message: 'DB error',
        source: 'api',
        timestamp: new Date('2026-01-01T00:01:00.000Z'),
        user_id: null,
      },
    ]);

    expect(body.pagination).toEqual({
      page: 1,
      pageSize: 50,
      total: 2,
      totalPages: 1,
      hasNext: false,
      hasPrev: false,
    });

    expect(dbConnect).toHaveBeenCalled();
  });

  test('applies a level filter to the MongoDB query', async () => {
    await GET(
      makeRequest('http://localhost:3000/api/logs?level=error'),
    );

    expect(Log.find).toHaveBeenCalledWith({
      level: 'error',
    });
  });

  test('maps a legacy user ID to its MongoDB ObjectId', async () => {
    const mongoUserId = new Types.ObjectId();

    userFindOneLean.mockResolvedValue({
      _id: mongoUserId,
    });

    await GET(
      makeRequest('http://localhost:3000/api/logs?user_id=42'),
    );

    expect(User.findOne).toHaveBeenCalledWith({
      legacy_id: '42',
    });

    expect(Log.find).toHaveBeenCalledWith({
      user_id: mongoUserId,
    });
  });

  test('returns no matches when a legacy user ID cannot be resolved', async () => {
    userFindOneLean.mockResolvedValue(null);

    await GET(
      makeRequest('http://localhost:3000/api/logs?user_id=99999'),
    );

    expect(Log.find).toHaveBeenCalledWith({
      user_id: {
        $in: [],
      },
    });
  });

  test('returns 500 when the MongoDB connection fails', async () => {
    (dbConnect as jest.Mock).mockRejectedValue(
      new Error('Connection refused'),
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
  });
  
  test('caps pageSize at 200', async () => {
  await GET(
    makeRequest(
      'http://localhost:3000/api/logs?pageSize=500',
    ),
  );

  expect(findChain.limit).toHaveBeenCalledWith(200);
  });

  test('sorts by timestamp descending by default', async () => {
    await GET(makeRequest());

    expect(findChain.sort).toHaveBeenCalledWith({
      timestamp: -1,
    });
  });

test('falls back to timestamp for an invalid sort field', async () => {
  await GET(
    makeRequest(
      'http://localhost:3000/api/logs?sortBy=username',
    ),
  );

  expect(findChain.sort).toHaveBeenCalledWith({
     timestamp: -1,
    });
  });

});

describe('DELETE /api/logs', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
    (dbConnect as jest.Mock).mockResolvedValue(undefined);
    (Log.deleteMany as jest.Mock).mockResolvedValue({
      deletedCount: 3,
    });
  });

  test('returns 401 for an unauthenticated request', async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const response = await DELETE(
      makeRequest(
        'http://localhost:3000/api/logs?olderThanDays=30',
      ),
    );

    expect(response.status).toBe(401);
    expect(Log.deleteMany).not.toHaveBeenCalled();
  });

  test('returns 400 when olderThanDays is invalid', async () => {
    const response = await DELETE(
      makeRequest(
        'http://localhost:3000/api/logs?olderThanDays=0',
      ),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);
    expect(Log.deleteMany).not.toHaveBeenCalled();
  });

  test('deletes logs older than the requested number of days', async () => {
    const response = await DELETE(
      makeRequest(
        'http://localhost:3000/api/logs?olderThanDays=30',
      ),
    );

    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.deleted).toBe(3);

    expect(Log.deleteMany).toHaveBeenCalledWith({
      timestamp: {
        $lt: expect.any(Date),
      },
    });
  });

  test('returns 500 when MongoDB deletion fails', async () => {
    (Log.deleteMany as jest.Mock).mockRejectedValue(
      new Error('Delete failed'),
    );

    const response = await DELETE(
      makeRequest(
        'http://localhost:3000/api/logs?olderThanDays=30',
      ),
    );

    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
  });
});