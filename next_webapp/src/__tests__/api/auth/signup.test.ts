/**
 * @jest-environment node
 *
 * Parity tests for the MongoDB /api/auth/signup routes.
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
        findOne: jest.fn(),
        create: jest.fn(),
    },
}));

jest.mock('@/models/mongoose/Role', () => ({
    __esModule: true,
    default: {
        findOne: jest.fn(),
    },
}));

jest.mock('bcryptjs', () => ({
    hash: jest.fn(),
}));

jest.mock('@/utils/logger', () => ({
    __esModule: true,
    default: {
        error: jest.fn(),
    },
}));

// ==============================
// Imports
// ==============================

import { POST } from '@/app/api/auth/signup/route';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import Role from '@/models/mongoose/Role';
import bcrypt from 'bcryptjs';

// ==============================
// Tests
// ==============================

describe('POST /api/auth/signup', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    const makeRequest = (body: unknown) =>
        new Request('http://localhost/api/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

    test('registers a new user successfully', async () => {
        (User.findOne as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(null),
        });

        (Role.findOne as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue({
                _id: 'role-object-id',
                legacy_id: '2',
                role_name: 'User',
            }),
        });

        (bcrypt.hash as jest.Mock).mockResolvedValue('hashed-password');

        (User.create as jest.Mock).mockResolvedValue({
            _id: 'user-id',
        });

        const response = await POST(
            makeRequest({
                firstName: 'Josh',
                lastName: 'Smith',
                email: ' Josh@Example.COM ',
                password: 'Password123!',
            })
        );

        const json = await response.json();

        expect(response.status).toBe(201);

        expect(json).toEqual({
            success: true,
            message: 'User registered successfully',
        });

        expect(dbConnect).toHaveBeenCalled();

        expect(User.findOne).toHaveBeenCalledWith({
            email: 'josh@example.com',
        });

        expect(bcrypt.hash).toHaveBeenCalledWith(
            'Password123!',
            10
        );

        expect(User.create).toHaveBeenCalledWith(
            expect.objectContaining({
                email: 'josh@example.com',
                password: 'hashed-password',
                role: {
                    id: 'role-object-id',
                    legacy_id: '2',
                    role_name: 'User',
                },
                profile: {
                    first_name: 'Josh',
                    last_name: 'Smith',
                },
            })
        );
    });

    test('returns 400 when required fields are missing', async () => {
        const response = await POST(
            makeRequest({
                firstName: 'Josh',
                lastName: 'Smith',
                email: 'josh@example.com',
            })
        );

        const json = await response.json();

        expect(response.status).toBe(400);
        expect(json.success).toBe(false);
        expect(json.message).toBe('All fields are required');

        expect(User.findOne).not.toHaveBeenCalled();
        expect(User.create).not.toHaveBeenCalled();
    });

    test('returns 400 when the user already exists', async () => {
        (User.findOne as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue({
                _id: 'existing-user',
            }),
        });

        const response = await POST(
            makeRequest({
                firstName: 'Josh',
                lastName: 'Smith',
                email: 'josh@example.com',
                password: 'Password123!',
            })
        );

        const json = await response.json();

        expect(response.status).toBe(400);
        expect(json.success).toBe(false);
        expect(json.message).toBe('User already exists');

        expect(bcrypt.hash).not.toHaveBeenCalled();
        expect(User.create).not.toHaveBeenCalled();
    });

    test('returns 500 when the default role cannot be found', async () => {
        (User.findOne as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(null),
        });

        (Role.findOne as jest.Mock).mockReturnValue({
            exec: jest.fn().mockResolvedValue(null),
        });

        const response = await POST(
            makeRequest({
                firstName: 'Josh',
                lastName: 'Smith',
                email: 'josh@example.com',
                password: 'Password123!',
            })
        );

        const json = await response.json();

        expect(response.status).toBe(500);
        expect(json.success).toBe(false);
        expect(json.message).toBe('Internal Server Error');

        expect(User.create).not.toHaveBeenCalled();
    });
});