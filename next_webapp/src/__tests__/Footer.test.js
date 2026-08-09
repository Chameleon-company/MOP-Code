/**
 * @jest-environment jsdom
 */

// Footer.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

jest.mock('next-intl', () => ({
  useTranslations: () => (key) => key,
}));

jest.mock('@/i18n-navigation', () => ({
  Link: ({ children, href, ...props }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

import Footer from '../components/Footer'; // Update the path to Footer.tsx



describe('Footer component', () => {
  test('renders footer with correct content', () => {
    render(<Footer />);

    // Check if the logo image is rendered
    const logoImage = screen.getByAltText('Melbourne Open Playground logo');
    expect(logoImage).toBeInTheDocument();

    // Check if the footer text content is rendered
    const footerText = screen.getByText(/Melbourne Open Playground/i);
    expect(footerText).toBeInTheDocument();

    const privacyPolicyLink = screen.getByText(/Privacy Policy/i);
    expect(privacyPolicyLink).toBeInTheDocument();

    const licensingLink = screen.getByText(/Licensing/i);
    expect(licensingLink).toBeInTheDocument();

    const contactLink = screen.getByText(/Contact Us/i);
    expect(contactLink).toBeInTheDocument();

    // Check if the copyright text is rendered
    const copyrightText = screen.getByText(
  new RegExp(`© ${new Date().getFullYear()}`, 'i')
);
    expect(copyrightText).toBeInTheDocument();
  });
});