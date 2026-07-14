/**
 * @jest-environment node
 *
 * Tests for DatabaseTransport
 * Verifies that log entries are correctly shaped and inserted into Supabase,
 * that extra fields land in meta, and that insert failures do not throw.
 */

// ─── Mocks ──────────────────────────────────────────────────────────────────

jest.mock('@/library/supabaseClient', () => ({
  supabase: { from: jest.fn() },
}));

// ─── Imports ─────────────────────────────────────────────────────────────────

import DatabaseTransport from '../../utils/databaseTransport';
import { supabase } from '@/library/supabaseClient';

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('DatabaseTransport', () => {
  let transport: DatabaseTransport;
  let mockInsert: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    transport = new DatabaseTransport({ level: 'info' });
    mockInsert = jest.fn().mockResolvedValue({ error: null });
    (supabase.from as jest.Mock).mockReturnValue({ insert: mockInsert });
  });

  test('inserts a row with the correct standard fields', async () => {
    const callback = jest.fn();

    await transport.log(
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

    expect(supabase.from).toHaveBeenCalledWith('logs');
    expect(mockInsert).toHaveBeenCalledWith([
      expect.objectContaining({
        level: 'info',
        message: 'User logged in',
        source: 'api',
        user_id: 42,
        method: 'POST',
        url: '/api/auth/login',
        status_code: 200,
        timestamp: expect.any(String),
      }),
    ]);
    expect(callback).toHaveBeenCalled();
  });

  test('captures unknown fields (e.g. error_code) in meta and does not drop them', async () => {
    const callback = jest.fn();

    await transport.log(
      {
        level: 'error',
        message: 'Login failed',
        source: 'api',
        status_code: 401,
        error_code: 'INVALID_CREDENTIALS',
      },
      callback,
    );

    expect(mockInsert).toHaveBeenCalledWith([
      expect.objectContaining({
        meta: { error_code: 'INVALID_CREDENTIALS' },
      }),
    ]);
    expect(callback).toHaveBeenCalled();
  });

  test('meta is empty object when all provided fields are standard columns', async () => {
    const callback = jest.fn();

    await transport.log(
      {
        level: 'info',
        message: 'Request received',
        source: 'middleware',
        user_id: 1,
        method: 'GET',
        url: '/api/profile',
      },
      callback,
    );

    expect(mockInsert).toHaveBeenCalledWith([
      expect.objectContaining({ meta: {} }),
    ]);
  });

  test('Supabase insert failure is caught and console.error-ed without throwing', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    mockInsert.mockResolvedValue({ error: { message: 'Connection refused' } });
    const callback = jest.fn();

    await transport.log({ level: 'error', message: 'something failed' }, callback);

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Failed to insert log into database:',
      expect.objectContaining({ message: 'Connection refused' }),
    );
    expect(callback).toHaveBeenCalled();

    consoleErrorSpy.mockRestore();
  });
});
