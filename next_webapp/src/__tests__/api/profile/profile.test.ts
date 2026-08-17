/**
 * @jest-environment node
 *
 * Tests for GET and PUT /api/profile.
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

jest.mock('@/models/mongoose/User', () => ({
    __esModule: true,
    default: {
        findById: jest.fn(),
    },
}));

import { GET, PUT } from '../../../app/api/profile/route';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';

const USER_ID = '507f1f77bcf86cd799439011';
const NEW_USER_ID = '507f1f77bcf86cd799439012';

function makeRequest(body?: unknown, userId?: string) {
    const headers = new Map<string, string>();

    if (userId) {
        headers.set('x-user-id', userId);
    }

    return {
        headers: {
            get: (key: string) => headers.get(key) ?? null,
        },
        json: jest.fn().mockResolvedValue(body ?? {}),
    } as any;
}

function makeUser(overrides: Record<string, unknown> = {}) {
    const user: any = {
        _id: USER_ID,
        email: 'jason@gmail.com',
        profile: {
            first_name: 'Jason',
            last_name: 'Holder',
            age: null,
            gender: null,
            profile_img: null,
            updated_at: null,
        },
        created_at: new Date('2026-03-22T11:09:32.253Z'),
        updated_at: new Date('2026-03-22T11:09:32.253Z'),
        ...overrides,
    };

    user.set = jest.fn((path: string, value: unknown) => {
        if (path === 'profile') {
            user.profile = value;
            return;
        }

        if (path.startsWith('profile.')) {
            if (!user.profile) {
                user.profile = {};
            }

            const field = path.replace('profile.', '');
            user.profile[field] = value;
            return;
        }

        user[path] = value;
    });

    user.save = jest.fn().mockResolvedValue(user);

    return user;
}

function mockFindByIdResult(result: unknown) {
    const chain: any = {
        exec: jest.fn().mockResolvedValue(result),
    };

    chain.select = jest.fn().mockReturnValue(chain);

    (User.findById as jest.Mock).mockReturnValue(chain);

    return chain;
}

beforeEach(() => {
    jest.clearAllMocks();
    (dbConnect as jest.Mock).mockResolvedValue(undefined);
    mockFindByIdResult(makeUser());
});

describe('GET /api/profile', () => {
    test('valid user returns 200 with profile data and email', async () => {
        const response = await GET(makeRequest(undefined, USER_ID));
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.user_id).toBe(USER_ID);
        expect(body.data.first_name).toBe('Jason');
        expect(body.data.last_name).toBe('Holder');
        expect(body.data.email).toBe('jason@gmail.com');

        expect(dbConnect).toHaveBeenCalledTimes(1);
        expect(User.findById).toHaveBeenCalledWith(USER_ID);
    });

    test('no x-user-id header returns 401', async () => {
        const response = await GET(makeRequest());
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
        expect(dbConnect).not.toHaveBeenCalled();
    });

    test('invalid MongoDB user ID returns 401', async () => {
        const response = await GET(makeRequest(undefined, '9'));
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
        expect(User.findById).not.toHaveBeenCalled();
    });

    test('user without a profile returns an empty profile shell', async () => {
        mockFindByIdResult(
            makeUser({
                profile: null,
            }),
        );

        const response = await GET(makeRequest(undefined, USER_ID));
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.first_name).toBeNull();
        expect(body.data.last_name).toBeNull();
        expect(body.data.email).toBe('jason@gmail.com');
    });

    test('user not found returns 401', async () => {
        mockFindByIdResult(null);

        const response = await GET(makeRequest(undefined, USER_ID));
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
    });

    test('database failure returns 500', async () => {
        (dbConnect as jest.Mock).mockRejectedValue(
            new Error('Database unavailable'),
        );

        const response = await GET(makeRequest(undefined, USER_ID));
        const body = await response.json();

        expect(response.status).toBe(500);
        expect(body.success).toBe(false);
    });
});

describe('PUT /api/profile', () => {
    test('updates first name and last name', async () => {
        const user = makeUser();
        mockFindByIdResult(user);

        const response = await PUT(
            makeRequest(
                {
                    first_name: 'Jason',
                    last_name: 'Smith',
                },
                USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.message).toBe('Profile updated successfully');
        expect(body.data.first_name).toBe('Jason');
        expect(body.data.last_name).toBe('Smith');
        expect(user.save).toHaveBeenCalledTimes(1);
    });

    test('updates email', async () => {
        const user = makeUser();
        mockFindByIdResult(user);

        const response = await PUT(
            makeRequest(
                {
                    email: 'newemail@gmail.com',
                },
                USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(user.set).toHaveBeenCalledWith(
            'email',
            'newemail@gmail.com',
        );
        expect(user.save).toHaveBeenCalledTimes(1);
    });

    test('updates age and gender', async () => {
        const user = makeUser();
        mockFindByIdResult(user);

        const response = await PUT(
            makeRequest(
                {
                    age: 25,
                    gender: 'Male',
                },
                USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.age).toBe(25);
        expect(body.data.gender).toBe('Male');
    });

    test('invalid gender returns 400 validation error', async () => {
        const response = await PUT(
            makeRequest(
                {
                    gender: 'Batman',
                },
                USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('gender');
        expect(dbConnect).not.toHaveBeenCalled();
    });

    test('age supplied as a string returns 400', async () => {
        const response = await PUT(
            makeRequest(
                {
                    age: '23',
                },
                USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.errors[0].field).toBe('age');
    });

    test('empty body returns 400', async () => {
        const response = await PUT(
            makeRequest({}, USER_ID),
        );
        const body = await response.json();

        expect(response.status).toBe(400);
        expect(body.success).toBe(false);
        expect(User.findById).not.toHaveBeenCalled();
    });

    test('no x-user-id header returns 401', async () => {
        const response = await PUT(
            makeRequest(
                {
                    first_name: 'Jason',
                },
                undefined,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
        expect(dbConnect).not.toHaveBeenCalled();
    });

    test('creates an embedded profile when one does not exist', async () => {
        const user = makeUser({
            _id: NEW_USER_ID,
            profile: null,
        });

        mockFindByIdResult(user);

        const response = await PUT(
            makeRequest(
                {
                    first_name: 'New',
                    last_name: 'User',
                },
                NEW_USER_ID,
            ),
        );
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(user.set).toHaveBeenNthCalledWith(
            1,
            'profile',
            expect.any(Object),
        );
        expect(body.data.first_name).toBe('New');
        expect(body.data.last_name).toBe('User');
        expect(user.save).toHaveBeenCalledTimes(1);
    });
});