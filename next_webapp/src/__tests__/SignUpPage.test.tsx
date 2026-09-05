/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import SignUpPage from '../app/[locale]/signup/page';

const mockPush = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

jest.mock('next-intl', () => ({
  useLocale: () => 'en',
}));

jest.mock('@/i18n-navigation', () => ({
  Link: ({ href, children, ...props }: any) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

describe('SignUpPage', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockPush.mockClear();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    jest.resetAllMocks();
  });

  test('shows validation errors for empty required fields', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    render(<SignUpPage />);

    await user.click(screen.getByRole('button', { name: /sign up/i }));

    expect(screen.getByText('Please enter your first name.')).toBeInTheDocument();
    expect(screen.getByText('Please enter your last name.')).toBeInTheDocument();
    expect(screen.getByText('Please enter a valid email address.')).toBeInTheDocument();
    expect(screen.getByText('Your password must be at least 8 characters long.')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('rejects a weak password', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    render(<SignUpPage />);

    await user.type(screen.getByLabelText('First Name'), 'Jane');
    await user.type(screen.getByLabelText('Last Name'), 'Doe');
    await user.type(screen.getByLabelText('Email'), 'jane@example.com');
    await user.type(screen.getByLabelText('Password'), 'weak');
    await user.type(screen.getByLabelText('Confirm Password'), 'weak');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    expect(screen.getByText('Your password must be at least 8 characters long.')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('rejects mismatched confirmation passwords', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    render(<SignUpPage />);

    await user.type(screen.getByLabelText('First Name'), 'Jane');
    await user.type(screen.getByLabelText('Last Name'), 'Doe');
    await user.type(screen.getByLabelText('Email'), 'jane@example.com');
    await user.type(screen.getByLabelText('Password'), 'StrongPass1!');
    await user.type(screen.getByLabelText('Confirm Password'), 'DifferentPass1!');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    expect(screen.getByText('Passwords do not match.')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('accepts a valid form, submits it, and redirects after success', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });
    global.fetch = mockFetch as jest.Mock;

    render(<SignUpPage />);

    await user.type(screen.getByLabelText('First Name'), 'Jane');
    await user.type(screen.getByLabelText('Last Name'), 'Doe');
    await user.type(screen.getByLabelText('Email'), 'jane@example.com');
    await user.type(screen.getByLabelText('Password'), 'StrongPass1!');
    await user.type(screen.getByLabelText('Confirm Password'), 'StrongPass1!');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));

    expect(mockFetch).toHaveBeenCalledWith(
      '/api/auth/signup',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          firstName: 'Jane',
          lastName: 'Doe',
          email: 'jane@example.com',
          password: 'StrongPass1!',
        }),
      }),
    );

    await waitFor(() => {
      expect(screen.getByText('Account created! Redirecting to login...')).toBeInTheDocument();
    });

    jest.advanceTimersByTime(1500);
    expect(mockPush).toHaveBeenCalledWith('/en/login');
  });

  test('shows an API error when signup fails', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const mockFetch = jest.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ message: 'Email already exists' }),
    });
    global.fetch = mockFetch as jest.Mock;

    render(<SignUpPage />);

    await user.type(screen.getByLabelText('First Name'), 'John');
    await user.type(screen.getByLabelText('Last Name'), 'Smith');
    await user.type(screen.getByLabelText('Email'), 'john@example.com');
    await user.type(screen.getByLabelText('Password'), 'StrongPass1!');
    await user.type(screen.getByLabelText('Confirm Password'), 'StrongPass1!');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    expect(await screen.findByText('Email already exists')).toBeInTheDocument();
  });
});
