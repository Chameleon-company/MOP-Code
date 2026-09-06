/**
 * @jest-environment node
 *
 * Parity tests for the MongoDB /api/auth/session route.
 * All MongoDB/Mongoose calls are mocked — no real DB work happens here.
 */

// ==============================
// Mocks — must come before imports
// ==============================

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

// ==============================
// Imports
// ==============================

import { GET } from '@/app/api/auth/session/route';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';

// ==============================
// Tests
// ==============================

describe('GET /api/auth/session', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    const makeRequest = (userId?: string) => {
        const headers = new Headers();

        if (userId !== undefined) {
            headers.set('x-user-id', userId);
        }

        return new Request(
            'http://localhost/api/auth/session',
            {
                method: 'GET',
                headers,
            },
        ) as any;
    };

    test('returns the current user session successfully', async () => {
        const userId = '507f1f77bcf86cd799439011';

        const mockUser = {
            _id: userId,
            email: 'josh@example.com',
            profile: {
                first_name: 'Josh',
                last_name: 'Smith',
                age: 21,
                gender: 'Male',
                profile_img: 'profile.jpg',
            },
            role: {
                legacy_id: '2',
                role_name: 'User',
            },
        };

        (User.findById as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(mockUser),
        });

        const response = await GET(makeRequest(userId));
        const json = await response.json();

        expect(response.status).toBe(200);

        expect(json).toEqual({
            success: true,
            message: 'Session retrieved successfully',
            data: {
                userId,
                email: 'josh@example.com',
                firstName: 'Josh',
                lastName: 'Smith',
                age: 21,
                gender: 'Male',
                profileImg: 'profile.jpg',
                roleId: 2,
                roleName: 'User',
            },
        });

        expect(dbConnect).toHaveBeenCalled();

        expect(User.findById).toHaveBeenCalledWith(userId);
    });

    test('returns 401 when x-user-id is missing', async () => {
        const response = await GET(makeRequest());
        const json = await response.json();

        expect(response.status).toBe(401);

        expect(json.success).toBe(false);
        expect(json.message).toBe('Unauthorised');

        expect(User.findById).not.toHaveBeenCalled();
    });

    test('returns 401 when x-user-id is invalid', async () => {
        const response = await GET(
            makeRequest('not-a-valid-object-id'),
        );

        const json = await response.json();

        expect(response.status).toBe(401);

        expect(json.success).toBe(false);
        expect(json.message).toBe('Unauthorised');

        expect(User.findById).not.toHaveBeenCalled();
    });

    test('returns 401 when the user does not exist', async () => {
        const userId = '507f1f77bcf86cd799439011';

        (User.findById as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(null),
        });

        const response = await GET(makeRequest(userId));
        const json = await response.json();

        expect(response.status).toBe(401);

        expect(json.success).toBe(false);
        expect(json.message).toBe('Unauthorised');

        expect(User.findById).toHaveBeenCalledWith(userId);
    });

    test('returns null for optional profile fields when they are missing', async () => {
        const userId = '507f1f77bcf86cd799439011';

        const mockUser = {
            _id: userId,
            email: 'josh@example.com',
            profile: {},
            role: {
                legacy_id: '2',
                role_name: 'User',
            },
        };

        (User.findById as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(mockUser),
        });

        const response = await GET(makeRequest(userId));
        const json = await response.json();

        expect(response.status).toBe(200);

        expect(json.data).toEqual({
            userId,
            email: 'josh@example.com',
            firstName: null,
            lastName: null,
            age: null,
            gender: null,
            profileImg: null,
            roleId: 2,
            roleName: 'User',
        });
    });

    test('returns 500 when the database throws an error', async () => {
        const userId = '507f1f77bcf86cd799439011';

        (User.findById as jest.Mock).mockReturnValue({
            exec: jest.fn().mockRejectedValue(
                new Error('Database error'),
            ),
        });

        const response = await GET(makeRequest(userId));
        const json = await response.json();

        expect(response.status).toBe(500);

        expect(json.success).toBe(false);
        expect(json.message).toBe('Internal Server Error');
    });
});