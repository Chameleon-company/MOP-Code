/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import ContactUsSection from '../components/ContactUsSection';

describe('ContactUsSection', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('shows validation errors when required fields are empty', async () => {
    const user = userEvent.setup();
    render(<ContactUsSection />);

    await user.click(screen.getByRole('button', { name: /send message/i }));

    expect(screen.getByText('Full name is required')).toBeInTheDocument();
    expect(screen.getByText('Email is required')).toBeInTheDocument();
    expect(screen.getByText('Subject is required')).toBeInTheDocument();
    expect(screen.getByText('Message is required')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('validates email format and caps message length at 255 characters', async () => {
    const user = userEvent.setup();
    render(<ContactUsSection />);

    await user.type(screen.getByPlaceholderText('Enter your full name'), 'Alex Morgan');
    await user.type(screen.getByPlaceholderText('Enter your email address'), 'not-an-email');
    await user.type(screen.getByPlaceholderText('Enter subject'), 'Test subject');
    const messageInput = screen.getByPlaceholderText('Write your message here...');
    await user.type(messageInput, 'a'.repeat(260));

    await user.click(screen.getByRole('button', { name: /send message/i }));

    expect(screen.getByText('Please enter a valid email address')).toBeInTheDocument();
    expect(messageInput).toHaveValue('a'.repeat(255));
    expect(screen.getByText('255/255')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('submits a valid form and shows the success message', async () => {
    const user = userEvent.setup();
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });
    global.fetch = mockFetch as jest.Mock;

    render(<ContactUsSection />);

    await user.type(screen.getByPlaceholderText('Enter your full name'), 'Jane Doe');
    await user.type(screen.getByPlaceholderText('Enter your email address'), 'jane@example.com');
    await user.type(screen.getByPlaceholderText('Enter subject'), 'Partnership enquiry');
    await user.type(screen.getByPlaceholderText('Write your message here...'), 'Hello team, I would love to collaborate.');
    await user.click(screen.getByRole('button', { name: /send message/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));

    expect(mockFetch).toHaveBeenCalledWith(
      '/api/contact',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fullName: 'Jane Doe',
          email: 'jane@example.com',
          subject: 'Partnership enquiry',
          message: 'Hello team, I would love to collaborate.',
        }),
      }),
    );

    expect(await screen.findByText('Your message has been sent successfully.')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter your full name')).toHaveValue('');
  });

  test('shows the failure message when the API rejects the submission', async () => {
    const user = userEvent.setup();
    const mockFetch = jest.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ success: false, message: 'Server unavailable' }),
    });
    global.fetch = mockFetch as jest.Mock;

    render(<ContactUsSection />);

    await user.type(screen.getByPlaceholderText('Enter your full name'), 'Sam Lee');
    await user.type(screen.getByPlaceholderText('Enter your email address'), 'sam@example.com');
    await user.type(screen.getByPlaceholderText('Enter subject'), 'Support');
    await user.type(screen.getByPlaceholderText('Write your message here...'), 'Need help with a bug');
    await user.click(screen.getByRole('button', { name: /send message/i }));

    expect(await screen.findByText('Server unavailable')).toBeInTheDocument();
  });
});
