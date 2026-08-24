/**
 * @jest-environment node
 *
 * Tests for the MongoDB Winston database transport.
 */

jest.mock('@/lib/dbConnect', () => ({
  __esModule: true,
  default: jest.fn(),
}));

jest.mock('@/models/mongoose/Log', () => ({
  __esModule: true,
  default: {
    insertMany: jest.fn(),
  },
}));

jest.mock('@/models/mongoose/User', () => ({
  __esModule: true,
  default: {
    find: jest.fn(),
  },
}));

import { Types } from 'mongoose';
import DatabaseTransport from '../../utils/databaseTransport';
import dbConnect from '@/lib/dbConnect';
import Log from '@/models/mongoose/Log';
import User from '@/models/mongoose/User';

describe('DatabaseTransport', () => {
  let transport: DatabaseTransport;
  let userLean: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();

    transport = new DatabaseTransport({ level: 'info' });

    (dbConnect as jest.Mock).mockResolvedValue(undefined);
    (Log.insertMany as jest.Mock).mockResolvedValue([]);

    userLean = jest.fn().mockResolvedValue([]);

    (User.find as jest.Mock).mockReturnValue({
      select: jest.fn().mockReturnValue({
        lean: userLean,
      }),
    });
  });

  afterEach(() => {
    transport.close();
  });

  test('writes a correctly shaped log batch to MongoDB', async () => {
    const mongoUserId = new Types.ObjectId();

    userLean.mockResolvedValue([
      {
        _id: mongoUserId,
        legacy_id: '42',
      },
    ]);

    const callback = jest.fn();

    transport.log(
      {
        level: 'info',
        message: 'User logged in',
        source: 'api',
        user_id: 42,
        method: 'POST',
        url: '/api/auth/login',
        status_code: 200,
      },
      callback,
    );

    await transport.flush();

    expect(callback).toHaveBeenCalled();
    expect(dbConnect).toHaveBeenCalled();

    expect(User.find).toHaveBeenCalledWith({
      legacy_id: { $in: ['42'] },
    });

    expect(Log.insertMany).toHaveBeenCalledWith(
      [
        expect.objectContaining({
          level: 'info',
          message: 'User logged in',
          source: 'api',
          user_id: mongoUserId,
          method: 'POST',
          url: '/api/auth/login',
          status_code: 200,
          timestamp: expect.any(Date),
        }),
      ],
      { ordered: false },
    );
  });

  test('removes ANSI colour codes before writing logs', async () => {
    const callback = jest.fn();

    transport.log(
      {
        level: '\u001b[32minfo\u001b[39m',
        message: '\u001b[31mDatabase error\u001b[39m',
        source: 'api',
      },
      callback,
    );

    await transport.flush();

    expect(Log.insertMany).toHaveBeenCalledWith(
      [
        expect.objectContaining({
          level: 'info',
          message: 'Database error',
        }),
      ],
      { ordered: false },
    );
  });

  test('stores unknown fields inside meta', async () => {
    const callback = jest.fn();

    transport.log(
      {
        level: 'error',
        message: 'Login failed',
        source: 'api',
        error_code: 'INVALID_CREDENTIALS',
      },
      callback,
    );

    await transport.flush();

    expect(Log.insertMany).toHaveBeenCalledWith(
      [
        expect.objectContaining({
          meta: {
            error_code: 'INVALID_CREDENTIALS',
          },
        }),
      ],
      { ordered: false },
    );
  });

  test('accepts an existing MongoDB ObjectId without a user lookup', async () => {
    const mongoUserId = new Types.ObjectId().toString();
    const callback = jest.fn();

    transport.log(
      {
        level: 'info',
        message: 'MongoDB user request',
        user_id: mongoUserId,
      },
      callback,
    );

    await transport.flush();

    expect(User.find).not.toHaveBeenCalled();

    expect(Log.insertMany).toHaveBeenCalledWith(
      [
        expect.objectContaining({
          user_id: mongoUserId,
        }),
      ],
      { ordered: false },
    );
  });

  test('catches MongoDB insert failures without throwing', async () => {
    const consoleErrorSpy = jest
      .spyOn(console, 'error')
      .mockImplementation(() => { });

    (Log.insertMany as jest.Mock).mockRejectedValue(
      new Error('Connection refused'),
    );

    const callback = jest.fn();

    transport.log(
      {
        level: 'error',
        message: 'Something failed',
      },
      callback,
    );

    await expect(transport.flush()).resolves.toBeUndefined();

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Database transport flush error:',
      expect.objectContaining({
        message: 'Connection refused',
      }),
    );

    expect(callback).toHaveBeenCalled();

    consoleErrorSpy.mockRestore();
  });
});