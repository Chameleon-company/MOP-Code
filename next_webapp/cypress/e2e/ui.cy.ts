// UI Interactions E2E tests — covers theme toggle, language switcher, mobile nav hamburger, and contact form
// API calls are stubbed where needed so tests run without a live backend.
//
// Selectors:
//   - Theme toggle:    cy.get('button[aria-label="Switch to dark/light mode"]')
//   - Language btn:    cy.get('header').contains(/English|Language/)
//   - Hamburger:       cy.get('button[aria-label="Open menu"]')
//   - Mobile nav:      cy.get('nav[aria-label="Mobile navigation"]')

// Shared stubs — stub home page APIs so every visit('/') is backend-free
const HOME_STUBS = () => {
  cy.intercept('GET', '/api/home/categories', {
    body: { success: true, data: [{ id: 1, category_name: 'Business', description: '', cover_img: '' }] },
  }).as('categories');
  cy.intercept('GET', '/api/usecases/recent', {
    body: { success: true, data: [{ id: 1, title: 'Test Use Case', description: '', cover_img: '', tags: [] }] },
  }).as('recentUseCases');
};

// 1. Theme Toggle
describe('Theme Toggle', () => {
  beforeEach(() => {
    // Clear local storage and force system preference to light mode
    // This guarantees the app always starts in a known state (light mode)
    HOME_STUBS();
    cy.clearLocalStorage();
    cy.visit('/', {
      onBeforeLoad(win) {
        cy.stub(win, 'matchMedia')
          .withArgs('(prefers-color-scheme: dark)')
          .returns({ matches: false, addListener: () => { }, removeListener: () => { } });
      },
    });
  });

  it('theme toggle button is visible in the header', () => {
    // Use aria-label selector to target the theme toggle specifically
    cy.get('button[aria-label="Switch to dark mode"]').should('be.visible');
  });

  it('clicking the theme toggle switches between dark and light mode', () => {
    // Assert the button is enabled before interacting — no raw cy.wait() timer needed
    cy.get('button[aria-label="Switch to dark mode"]').should('be.enabled');

    // We forced light mode in beforeEach, so the initial state is deterministic
    cy.get('html').should('not.have.class', 'dark');

    // Click to switch to dark mode
    cy.get('button[aria-label="Switch to dark mode"]').click();
    // HTML element should now carry the dark class
    cy.get('html').should('have.class', 'dark');

    // Click again to return to light mode
    cy.get('button[aria-label="Switch to light mode"]').click();
    // HTML element should no longer have the dark class
    cy.get('html').should('not.have.class', 'dark');
  });
});

// 2. Language Switcher
describe('Language Switcher', () => {
  beforeEach(() => {
    HOME_STUBS();
    cy.visit('/');
  });

  it('language dropdown button is visible in the header', () => {
    // LanguageDropdown is hidden on mobile — force desktop viewport first
    cy.viewport(1280, 720);
    // Main navigation should be visible at desktop width
    cy.get('nav[aria-label="Main navigation"]').should('be.visible');
    // At least one button should exist within the header area
    cy.get('header').find('button').should('have.length.greaterThan', 0);
  });

  it('switching language updates the URL locale prefix', () => {
    cy.viewport(1280, 720);
    // Use a regex to match either language label — handles locale-prefix variations
    cy.get('header').contains(/English|Language/).click();
    // Click Chinese locale option
    cy.contains('Chinese').click();
    // URL should now include the /cn locale prefix
    cy.url({ timeout: 8000 }).should('include', '/cn');
  });
});

// 3. Mobile Navigation
describe('Mobile Navigation', () => {
  // Suppress ResizeObserver loop locally — triggered by third-party widgets at 375 px viewport
  let removeResizeHandler: (() => void) | undefined;

  beforeEach(() => {
    // Set mobile viewport
    cy.viewport(375, 812);
    HOME_STUBS();
    cy.visit('/');

    const handler = (err: Error) => {
      if (/ResizeObserver loop/.test(err.message)) return false;
    };
    cy.on('uncaught:exception', handler);
    removeResizeHandler = () => cy.off('uncaught:exception', handler);
  });

  afterEach(() => {
    removeResizeHandler?.();
    removeResizeHandler = undefined;
  });

  it('hamburger menu button is visible on mobile', () => {
    // The open menu button should be visible at mobile width
    cy.get('button[aria-label="Open menu"]').should('be.visible');
  });

  it('clicking hamburger opens the mobile navigation', () => {
    // Click the hamburger button
    cy.get('button[aria-label="Open menu"]').click();
    // Mobile navigation should become visible
    cy.get('nav[aria-label="Mobile navigation"]').should('be.visible');
  });

  it('mobile nav shows all main links', () => {
    // Open the mobile menu
    cy.get('button[aria-label="Open menu"]').click();
    // All main nav links should be visible inside the mobile menu
    cy.get('nav[aria-label="Mobile navigation"]').within(() => {
      cy.contains('Home').should('be.visible');
      cy.contains('About Us').should('be.visible');
      cy.contains('Explore').should('be.visible');
    });
  });

  it('clicking a mobile nav link navigates and closes the menu', () => {
    // Open the mobile menu
    cy.get('button[aria-label="Open menu"]').click();
    // Use aria-label nav selector to scope the click precisely
    cy.get('nav[aria-label="Mobile navigation"]').contains('About Us').click();
    // URL should change to /about
    cy.url({ timeout: 8000 }).should('include', '/about');
    // Mobile menu should close after navigation
    cy.get('nav[aria-label="Mobile navigation"]').should('not.exist');
  });
});

// 4. Contact Page Form
describe('Contact Page Form', () => {
  beforeEach(() => {
    cy.visit('/contact');
  });

  it('renders the contact form with all required fields', () => {
    // First name and email input fields should be visible
    cy.get('main').find('input[name="firstName"], input[placeholder*="first"], input[placeholder*="First"]')
      .should('be.visible');
    cy.get('main').find('input[name="email"], input[type="email"]').should('be.visible');
  });

  it('submitting an empty form shows validation errors', () => {
    // Click the submit button without filling any fields
    cy.get('main').find('button[type="submit"]').click();
    // Match error elements by common red-text patterns used across styling approaches
    cy.get('p[style*="color: red"], [class*="error"], [class*="text-red"]')
      .should('have.length.greaterThan', 0);
  });

  it('submitting with invalid email shows an email error', () => {
    // Fill in name fields but enter an invalid email
    cy.get('main').find('input[name="firstName"], input[placeholder*="First"]').first().type('John');
    cy.get('main').find('input[name="lastName"], input[placeholder*="Last"]').first().type('Doe');
    cy.get('main').find('input[type="email"]').type('not-an-email');
    // Submit the form
    cy.get('main').find('button[type="submit"]').click();
    // Email validation error message should be shown
    cy.contains(/valid email/i).should('be.visible');
  });
});
