/**
 * @jest-environment node
 *
 * Tests for PUT /api/usecases.
 * MongoDB, GridFS and authentication dependencies are mocked.
 */

// ─── Mocks ──────────────────────────────────────────────────────────────────

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
    findById: jest.fn(),
  },
}));

jest.mock('@/app/api/library/auth', () => ({
  getAuthUser: jest.fn(),
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

jest.mock('../../../app/api/usecases/_shared', () => {
  class InvalidCategoryRefError extends Error {
    constructor(message?: string) {
      super(message);
      this.name = 'InvalidCategoryRefError';
    }
  }

  return {
    InvalidCategoryRefError,
    resolveCategoryRef: jest.fn(),
    resolveCreatedBy: jest.fn(),
    resolveTagRefs: jest.fn(),
    validateNotebookContent: jest.fn(),
    uploadNotebookToGridFS: jest.fn(),
    tryDeleteGridFSFile: jest.fn()
  };
});

// ─── Imports ────────────────────────────────────────────────────────────────

import { PUT } from '../../../app/api/usecases/[id]/route';
import UseCase from '@/models/mongoose/UseCase';
import { getAuthUser } from '@/app/api/library/auth';
import { InvalidCategoryRefError, resolveCategoryRef, resolveTagRefs } from '@/app/api/usecases/_shared';
import { mock } from 'node:test';

// ─── Helpers ────────────────────────────────────────────────────────────────

const ADMIN_ID = '108fbc1f77a0123ff5026cf1';
const USE_CASE_ID = '108fbc1f77a0123ff5026cf1';

function makeRequest(body: unknown) {
    return { json: jest.fn().mockResolvedValue(body), headers: { get: jest.fn().mockReturnValue(null) }} as any;
};

// PUT reads the id via 'await params'
function makeRouteContext(id: string) {
    return { params: Promise.resolve({ id })} as any;
};

// create a fake 'existing' Mongoose page
function mockExisting() {
    const existing: any = {
        content_file_id: null,
        category: { id: null, legacy_id: '6', category_name: null},
        set: jest.fn(),
        save: jest.fn().mockResolvedValue({ toObject: () => ({}) })
    };
    (UseCase.findById as jest.Mock).mockResolvedValue(existing);
    return existing;
};

// ─── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
    jest.clearAllMocks();
    (getAuthUser as jest.Mock).mockReturnValue({
        userId: ADMIN_ID,
        isAuthenticated: true,
        isAdmin: true
    });
    (resolveCategoryRef as jest.Mock).mockResolvedValue(null);
    (resolveTagRefs as jest.Mock).mockResolvedValue([]);
});

describe('PUT /api/usecases/[id] - category_id', () => {
    test('invalid category_id returns 400 and does not save', async() => {
        const existing = mockExisting();
        (resolveCategoryRef as jest.Mock).mockRejectedValue(new InvalidCategoryRefError('Invalid category_id: "NaN"'));

        const response = await PUT(
            makeRequest({ category_id: 'NaN', tags: ['new-tag'] }),
            makeRouteContext(USE_CASE_ID)
        );
        const body = await response.json();

        expect(response.status).toBe(400);
        expect(body.code).toBe('INVALID_CATEGORY_ID');
        // check that no tag is created and nothing is saved
        expect(resolveTagRefs).not.toHaveBeenCalled();
        expect(existing.save).not.toHaveBeenCalled();
    });

    test('a valid category ref is applied to the document', async() => {
        const existing = mockExisting();
        const ref = { id: null, legacy_id: '14', category_name: null };
        (resolveCategoryRef as jest.Mock).mockResolvedValue(ref);

        const response = await PUT(
            makeRequest({ category_id: '14'}),
            makeRouteContext(USE_CASE_ID)
        );

        expect(response.status).toBe(200);
        expect(resolveCategoryRef).toHaveBeenCalledWith('14');
        expect(existing.category).toEqual(ref);
        expect(existing.save).toHaveBeenCalled();
    });

    test('category_id: null clears the category', async() => {
        const existing = mockExisting();

        const response = await PUT(
            makeRequest({ category_id: null }),
            makeRouteContext(USE_CASE_ID)
        );

        expect(response.status).toBe(200);
        expect(resolveCategoryRef).toHaveBeenCalledWith(null);
        expect(existing.category).toBeNull();
    });

    test('non-existing category_id does not affect the category', async() => {
        const existing = mockExisting();

        const response = await PUT(
            makeRequest({ title: 'new title' }),
            makeRouteContext(USE_CASE_ID)
        );

        expect(response.status).toBe(200);
        expect(resolveCategoryRef).not.toHaveBeenCalled();
        expect(existing.category).toEqual({ id: null, legacy_id: '6', category_name: null });
    });
});
