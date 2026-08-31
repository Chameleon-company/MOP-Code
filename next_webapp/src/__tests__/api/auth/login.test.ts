/**
 * @jest-environment node
 *
 * Tests for POST /api/auth/login.
 * MongoDB, bcrypt and JWT operations are mocked.
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
        findOne: jest.fn(),
    },
}));

jest.mock('bcryptjs', () => ({
    compare: jest.fn(),
}));

jest.mock('jsonwebtoken', () => ({
    sign: jest.fn(),
}));

import { POST } from '../../../app/api/auth/login/route';
import dbConnect from '@/lib/dbConnect';
import User from '@/models/mongoose/User';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

function makeRequest(body: object) {
    return {
        json: jest.fn().mockResolvedValue(body),
    } as unknown as Request;
}

function mockFindOneResult(result: unknown) {
    (User.findOne as jest.Mock).mockReturnValue({
        exec: jest.fn().mockResolvedValue(result),
    });
}

const MOCK_USER = {
    _id: '507f1f77bcf86cd799439011',
    email: 'user@test.com',
    password: '$2a$10$hash',
    role: {
        legacy_id: '2',
        role_name: 'user',
    },
    profile: {
        first_name: 'Jane',
        last_name: 'Doe',
    },
};

const MOCK_TOKEN = 'mock.jwt.token';

beforeEach(() => {
    jest.clearAllMocks();

    (dbConnect as jest.Mock).mockResolvedValue(undefined);
    mockFindOneResult(MOCK_USER);
    (bcrypt.compare as jest.Mock).mockResolvedValue(true);
    (jwt.sign as jest.Mock).mockReturnValue(MOCK_TOKEN);
});

describe('POST /api/auth/login', () => {
    test('valid credentials returns 200 with token', async () => {
        const request = makeRequest({
            email: '  USER@TEST.COM ',
            password: 'password123',
        });

        const response = await POST(request);
        const body = await response.json();

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data.token).toBe(MOCK_TOKEN);
        expect(body.data.email).toBe(MOCK_USER.email);
        expect(body.data.firstName).toBe(MOCK_USER.profile.first_name);
        expect(body.data.lastName).toBe(MOCK_USER.profile.last_name);
        expect(body.data.roleId).toBe(2);

        expect(dbConnect).toHaveBeenCalledTimes(1);
        expect(User.findOne).toHaveBeenCalledWith({
            email: 'user@test.com',
        });
        expect(bcrypt.compare).toHaveBeenCalledWith(
            'password123',
            MOCK_USER.password,
        );
    });

    test('wrong password returns 401 INVALID_CREDENTIALS', async () => {
        (bcrypt.compare as jest.Mock).mockResolvedValue(false);

        const response = await POST(
            makeRequest({
                email: 'user@test.com',
                password: 'wrongpass',
            }),
        );
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INVALID_CREDENTIALS');
        expect(jwt.sign).not.toHaveBeenCalled();
    });

    test('unknown email returns 401 INVALID_CREDENTIALS', async () => {
        mockFindOneResult(null);

        const response = await POST(
            makeRequest({
                email: 'nobody@test.com',
                password: 'password123',
            }),
        );
        const body = await response.json();

        expect(response.status).toBe(401);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INVALID_CREDENTIALS');
        expect(bcrypt.compare).not.toHaveBeenCalled();
    });

    test('missing fields returns 400 MISSING_FIELDS', async () => {
        const response = await POST(
            makeRequest({
                email: 'user@test.com',
            }),
        );
        const body = await response.json();

        expect(response.status).toBe(400);
        expect(body.success).toBe(false);
        expect(body.code).toBe('MISSING_FIELDS');
        expect(User.findOne).not.toHaveBeenCalled();
    });

    test('database failure returns 500 INTERNAL_ERROR', async () => {
        (dbConnect as jest.Mock).mockRejectedValue(
            new Error('Connection refused'),
        );

        const response = await POST(
            makeRequest({
                email: 'user@test.com',
                password: 'password123',
            }),
        );
        const body = await response.json();

        expect(response.status).toBe(500);
        expect(body.success).toBe(false);
        expect(body.code).toBe('INTERNAL_ERROR');
    });
});