/**
 * @jest-environment node
 *
 * Tests for POST /api/usecases.
 * MongoDB, GridFS and authentication dependencies are mocked.
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
    create: jest.fn(),
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

jest.mock('../../../app/api/usecases/_shared', () => ({
  resolveCategoryRef: jest.fn(),
  resolveCreatedBy: jest.fn(),
  resolveTagRefs: jest.fn(),
  validateNotebookContent: jest.fn(),
  uploadNotebookToGridFS: jest.fn(),
  tryDeleteGridFSFile: jest.fn(),
}));

import { POST } from '../../../app/api/usecases/route';
import dbConnect from '@/lib/dbConnect';
import UseCase from '@/models/mongoose/UseCase';
import { getAuthUser } from '@/app/api/library/auth';
import {
  resolveCategoryRef,
  resolveCreatedBy,
  resolveTagRefs,
  validateNotebookContent,
  uploadNotebookToGridFS,
  tryDeleteGridFSFile,
} from '../../../app/api/usecases/_shared';

const ADMIN_ID = '507f1f77bcf86cd799439011';
const CREATOR_ID = '507f1f77bcf86cd799439012';
const CATEGORY_ID = '507f1f77bcf86cd799439013';
const TAG_ID = '507f1f77bcf86cd799439014';
const FILE_ID = '507f1f77bcf86cd799439015';

const ADMIN_AUTH = {
  userId: ADMIN_ID,
  isAuthenticated: true,
  isAdmin: true,
};

const CREATED_DOCUMENT = {
  _id: '507f1f77bcf86cd799439020',
  title: 'Test use case',
  description: 'A description',
  cover_img: null,
  content_file_id: null,
  content_type: null,
  category: null,
  tags: [],
  created_by: CREATOR_ID,
};

function makeRequest(body: unknown) {
  return {
    json: jest.fn().mockResolvedValue(body),
    headers: {
      get: jest.fn().mockReturnValue(null),
    },
  } as any;
}

function makeInvalidJsonRequest() {
  return {
    json: jest.fn().mockRejectedValue(
      new SyntaxError('Invalid JSON'),
    ),
    headers: {
      get: jest.fn().mockReturnValue(null),
    },
  } as any;
}

function makeCreatedModel(document = CREATED_DOCUMENT) {
  return {
    toObject: jest.fn().mockReturnValue(document),
  };
}

beforeEach(() => {
  jest.clearAllMocks();

  (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
  (dbConnect as jest.Mock).mockResolvedValue(undefined);

  (resolveCategoryRef as jest.Mock).mockResolvedValue(null);
  (resolveCreatedBy as jest.Mock).mockResolvedValue(
    CREATOR_ID,
  );
  (resolveTagRefs as jest.Mock).mockResolvedValue([]);

  (validateNotebookContent as jest.Mock).mockReturnValue({
    valid: true,
    contentType: 'notebook',
    notebookBuffer: Buffer.from('notebook'),
  });

  (uploadNotebookToGridFS as jest.Mock).mockResolvedValue(
    FILE_ID,
  );
  (tryDeleteGridFSFile as jest.Mock).mockResolvedValue(
    undefined,
  );

  (UseCase.create as jest.Mock).mockResolvedValue(
    makeCreatedModel(),
  );
});

describe('POST /api/usecases - authorization', () => {
  test('unauthenticated request returns 401', async () => {
    (getAuthUser as jest.Mock).mockReturnValue({
      userId: null,
      isAuthenticated: false,
      isAdmin: false,
    });

    const response = await POST(
      makeRequest({
        title: 'Test use case',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.code).toBe('UNAUTHORIZED');
    expect(dbConnect).not.toHaveBeenCalled();
    expect(UseCase.create).not.toHaveBeenCalled();
  });

  test('authenticated non-admin returns 403', async () => {
    (getAuthUser as jest.Mock).mockReturnValue({
      userId: ADMIN_ID,
      isAuthenticated: true,
      isAdmin: false,
    });

    const response = await POST(
      makeRequest({
        title: 'Test use case',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(403);
    expect(body.code).toBe('FORBIDDEN');
    expect(dbConnect).not.toHaveBeenCalled();
  });
});

describe('POST /api/usecases - request validation', () => {
  test('malformed JSON returns 400 INVALID_JSON', async () => {
    const response = await POST(
      makeInvalidJsonRequest(),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_JSON');
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('missing title returns 400 MISSING_FIELDS', async () => {
    const response = await POST(
      makeRequest({
        description: 'Missing title',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('MISSING_FIELDS');
    expect(dbConnect).not.toHaveBeenCalled();
  });

  test('whitespace title returns 400 MISSING_FIELDS', async () => {
    const response = await POST(
      makeRequest({
        title: '   ',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('MISSING_FIELDS');
    expect(UseCase.create).not.toHaveBeenCalled();
  });

  test('invalid notebook content returns its validation error', async () => {
    (validateNotebookContent as jest.Mock).mockReturnValue({
      valid: false,
      message: 'content is not valid notebook JSON',
      status: 400,
      code: 'INVALID_NOTEBOOK_JSON',
    });

    const response = await POST(
      makeRequest({
        title: 'Notebook case',
        content: 'not-json',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('INVALID_NOTEBOOK_JSON');
    expect(dbConnect).not.toHaveBeenCalled();
    expect(uploadNotebookToGridFS).not.toHaveBeenCalled();
  });
});

describe('POST /api/usecases - MongoDB creation', () => {
  test('creates a use case without notebook content', async () => {
    const response = await POST(
      makeRequest({
        title: '  Test use case  ',
        description: 'A description',
        created_by: 'client-supplied-id',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(201);
    expect(body.success).toBe(true);

    expect(dbConnect).toHaveBeenCalledTimes(1);
    expect(resolveCreatedBy).toHaveBeenCalledWith(
      ADMIN_ID,
    );

    expect(UseCase.create).toHaveBeenCalledWith({
      title: 'Test use case',
      description: 'A description',
      cover_img: null,
      content_file_id: null,
      content_type: null,
      category: null,
      tags: [],
      created_by: CREATOR_ID,
    });

    expect(uploadNotebookToGridFS).not.toHaveBeenCalled();
  });

  test('resolves category and embedded tags', async () => {
    const categoryRef = {
      id: CATEGORY_ID,
      legacy_id: '3',
      category_name: 'Technology',
    };

    const tagRefs = [
      {
        id: TAG_ID,
        legacy_id: null,
        name: 'Artificial Intelligence',
        slug: 'artificial-intelligence',
      },
    ];

    (resolveCategoryRef as jest.Mock).mockResolvedValue(
      categoryRef,
    );
    (resolveTagRefs as jest.Mock).mockResolvedValue(
      tagRefs,
    );

    const response = await POST(
      makeRequest({
        title: 'AI use case',
        category_id: '3',
        tags: ['Artificial Intelligence'],
      }),
    );

    expect(response.status).toBe(201);
    expect(resolveCategoryRef).toHaveBeenCalledWith('3');
    expect(resolveTagRefs).toHaveBeenCalledWith([
      'Artificial Intelligence',
    ]);

    expect(UseCase.create).toHaveBeenCalledWith(
      expect.objectContaining({
        category: categoryRef,
        tags: tagRefs,
      }),
    );
  });

  test('uploads validated notebook content to GridFS', async () => {
    const notebookBuffer = Buffer.from(
      '{"cells":[]}',
    );

    (validateNotebookContent as jest.Mock).mockReturnValue({
      valid: true,
      contentType: 'notebook',
      notebookBuffer,
    });

    const response = await POST(
      makeRequest({
        title: 'Notebook use case',
        content: '{"cells":[]}',
      }),
    );

    expect(response.status).toBe(201);
    expect(validateNotebookContent).toHaveBeenCalledWith(
      '{"cells":[]}',
    );
    expect(uploadNotebookToGridFS).toHaveBeenCalledWith(
      notebookBuffer,
    );

    expect(UseCase.create).toHaveBeenCalledWith(
      expect.objectContaining({
        content_file_id: FILE_ID,
        content_type: 'notebook',
      }),
    );
  });

  test('does not trust created_by supplied by the client', async () => {
    await POST(
      makeRequest({
        title: 'Secure creator test',
        created_by: 'spoofed-user-id',
      }),
    );

    expect(resolveCreatedBy).toHaveBeenCalledWith(
      ADMIN_ID,
    );
    expect(resolveCreatedBy).not.toHaveBeenCalledWith(
      'spoofed-user-id',
    );
  });
});

describe('POST /api/usecases - failures and cleanup', () => {
  test('failed save after GridFS upload cleans up the file', async () => {
    const saveError = new Error('MongoDB save failed');

    (UseCase.create as jest.Mock).mockRejectedValue(
      saveError,
    );

    const response = await POST(
      makeRequest({
        title: 'Notebook use case',
        content: '{"cells":[]}',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.code).toBe('INTERNAL_ERROR');

    expect(tryDeleteGridFSFile).toHaveBeenCalledWith(
      FILE_ID,
      'orphaned upload after failed create',
    );
  });

  test('Mongoose validation error returns 400', async () => {
    const validationError = new Error(
      'Use case validation failed',
    );
    validationError.name = 'ValidationError';

    (UseCase.create as jest.Mock).mockRejectedValue(
      validationError,
    );

    const response = await POST(
      makeRequest({
        title: 'Validation case',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.code).toBe('VALIDATION_ERROR');
  });

  test('database connection failure returns 500', async () => {
    (dbConnect as jest.Mock).mockRejectedValue(
      new Error('Database unavailable'),
    );

    const response = await POST(
      makeRequest({
        title: 'Database failure',
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.code).toBe('INTERNAL_ERROR');
    expect(UseCase.create).not.toHaveBeenCalled();
  });
});