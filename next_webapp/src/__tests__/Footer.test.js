// Footer.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Footer from '../components/Footer'; // Update the path to Footer.tsx
import Link from 'next/link';
// …
<li><Link href="/upload">Contribute Data</Link></li>



describe('Footer component', () => {
  test('renders footer with correct content', () => {
    render(<Footer />);

    // Check if the logo image is rendered
    const logoImage = screen.getByAltText('MOP logo');
    expect(logoImage).toBeInTheDocument();

    // Check if the footer text content is rendered
    const footerText = screen.getByText(/Melbourne Open Playground/i);
    expect(footerText).toBeInTheDocument();

    // Check if the links are rendered
    const aboutLink = screen.getByText(/About/i);
    expect(aboutLink).toBeInTheDocument();

    const privacyPolicyLink = screen.getByText(/Privacy Policy/i);
    expect(privacyPolicyLink).toBeInTheDocument();

    const licensingLink = screen.getByText(/Licensing/i);
    expect(licensingLink).toBeInTheDocument();

    const contactLink = screen.getByText(/Contact/i);
    expect(contactLink).toBeInTheDocument();

    // Check if the copyright text is rendered
    const copyrightText = screen.getByText(/© 2023/i);
    expect(copyrightText).toBeInTheDocument();

    //ensures news section is covered by automated tests
    const newsletterTitle = screen.getByText(/Subscribe to Our Newsletter/i);
    expect(newsletterTitle).toBeInTheDocument();

  


  });
});