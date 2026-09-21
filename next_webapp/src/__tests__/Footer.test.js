/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import Footer from '../components/Footer';

jest.mock('next-intl', () => ({
  useTranslations: () => (key) => ({
    'Footer intro': "Exploring Melbourne's open data to build smarter communities.",
    'Quick Links': 'Quick Links',
    Connect: 'Connect',
    'Follow us': 'Follow us',
    Newsletter: 'Newsletter',
    'Newsletter description': 'Get Melbourne open-data updates first.',
    'Enter your email': 'Enter your email',
    Submit: 'Submit',
    'Email for newsletter': 'Email for newsletter',
    'Newsletter required': 'Please enter your email address.',
    'Newsletter invalid': 'Please enter a valid email address (e.g. morgan.lee@gmail.com).',
    'Newsletter success': "You're in — we'll only email when there's something worth your time.",
    Copyright: 'Melbourne Open Playground. All rights reserved.',
    'accessibility.home': 'Go to home page',
  }[key] ?? key),
}));

jest.mock('@/i18n-navigation', () => ({
  Link: ({ href, children, ...props }) => {
    const React = require('react');

    return React.createElement(
      'a',
      { href, ...props },
      children,
    );
  },
}));

jest.mock('next/image', () => ({
  __esModule: true,
  default: (props) => {
    const React = require('react');

    return React.createElement('img', props);
  },
}));

beforeEach(() => {
  global.requestAnimationFrame = jest.fn(() => 1);
  global.cancelAnimationFrame = jest.fn();
});

describe('Footer component', () => {
  test('renders the logo and project description', () => {
    render(<Footer />);

    expect(
      screen.getByAltText(
        'Melbourne Open Playground logo',
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        /Exploring Melbourne's open data/i,
      ),
    ).toBeInTheDocument();
  });

  test('renders the current quick links', () => {
    render(<Footer />);

    expect(
      screen.getByRole('link', {
        name: 'Licensing',
      }),
    ).toHaveAttribute('href', '/licensing');

    expect(
      screen.getByRole('link', {
        name: 'Privacy Policy',
      }),
    ).toHaveAttribute('href', '/privacypolicy');

    expect(
      screen.getByRole('link', {
        name: 'Contact Us',
      }),
    ).toHaveAttribute('href', '/contact');
  });

  test('renders social and newsletter controls', () => {
    render(<Footer />);

    expect(
      screen.getByRole('link', {
        name: 'Facebook',
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole('link', {
        name: 'Twitter/X',
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole('link', {
        name: 'LinkedIn',
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByLabelText('Email for newsletter'),
    ).toBeInTheDocument();

    expect(
      screen.getByRole('button', {
        name: 'Submit',
      }),
    ).toBeInTheDocument();
  });

  test('renders the current copyright year', () => {
    render(<Footer />);

    const currentYear = new Date().getFullYear();

    expect(
      screen.getByText(
        new RegExp(
          `${currentYear} Melbourne Open Playground`,
          'i',
        ),
      ),
    ).toBeInTheDocument();
  });

  test('shows a validation error for an invalid newsletter email', async () => {
    const user = userEvent.setup();
    render(<Footer />);

    await user.type(
      screen.getByLabelText('Email for newsletter'),
      'not-an-email',
    );
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    expect(screen.getByRole('alert')).toHaveTextContent(
      /Please enter a valid email address/i,
    );
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  test('accepts a valid email and shows a success toast', async () => {
    const user = userEvent.setup();
    render(<Footer />);

    const input = screen.getByLabelText('Email for newsletter');

    await user.type(input, 'morgan.lee@gmail.com');
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    expect(input).toHaveValue('');
    expect(screen.getByRole('status')).toHaveTextContent(
      /we'll only email when there's something worth your time/i,
    );
  });
});