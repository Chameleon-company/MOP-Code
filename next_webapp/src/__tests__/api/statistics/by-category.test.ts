/**
 * @jest-environment node
 *
 * Parity tests for the MongoDB /api/statistics/by-category route.
 * All MongoDB/Mongoose calls are mocked — no real DB work happens here.
 */

// ==============================
// Mocks — must come before imports
// ==============================

jest.mock('@/lib/dbConnect', () => ({
  __esModule: true,
  default: jest.fn(),
}));

jest.mock('@/models/mongoose/UseCase', () => ({
  __esModule: true,
  default: {
    aggregate: jest.fn(),
  },
}));

// ==============================
// Imports
// ==============================

import { GET } from '@/app/api/statistics/by-category/route';
import dbConnect from '@/lib/dbConnect';
import UseCaseModel from '@/models/mongoose/UseCase';

const mockedDbConnect = dbConnect as jest.MockedFunction<typeof dbConnect>;
const mockedAggregate = UseCaseModel.aggregate as jest.Mock;

// ==============================
// Mock Data
// ==============================

const MOCK_RESULT = [
  {
    category: 'Healthcare',
    count: 5,
  },
  {
    category: 'Education',
    count: 3,
  },
  {
    category: 'Uncategorized',
    count: 1,
  },
];

// ==============================
// Setup
// ==============================

beforeEach(() => {
  jest.clearAllMocks();
  mockedDbConnect.mockResolvedValue(undefined as any);
  mockedAggregate.mockResolvedValue(MOCK_RESULT);
});

// ==============================
// Tests
// ==============================

describe('GET /api/statistics/by-category', () => {
  test('success, 200 returns category counts', async () => {
    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual(MOCK_RESULT);

    expect(dbConnect).toHaveBeenCalledTimes(1);
    expect(mockedAggregate).toHaveBeenCalledTimes(1);
  });

  test('success, aggregation pipeline groups by category with fallback and sorts by count descending', async () => {
    await GET();

    const pipeline = mockedAggregate.mock.calls[0][0];
    expect(pipeline).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          $group: expect.objectContaining({ _id: '$category.category_name' }),
        }),
        expect.objectContaining({
          $project: expect.objectContaining({
            category: { $ifNull: ['$_id', 'Uncategorized'] },
          }),
        }),
        expect.objectContaining({ $sort: { count: -1 } }),
      ])
    );
  });

  test('empty data, 200 with empty array', async () => {
    mockedAggregate.mockResolvedValue([]);

    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual([]);
  });

  test('aggregation error, 500 internal server error', async () => {
    mockedAggregate.mockRejectedValue(new Error('Database error'));

    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.message).toBe('Internal server error');

    expect(dbConnect).toHaveBeenCalledTimes(1);
    expect(mockedAggregate).toHaveBeenCalledTimes(1);
  });

  test('dbConnect error, 500 internal server error', async () => {
    mockedDbConnect.mockRejectedValue(new Error('Connection failed'));

    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(500);
    expect(body.success).toBe(false);
    expect(mockedAggregate).not.toHaveBeenCalled();
  });
});