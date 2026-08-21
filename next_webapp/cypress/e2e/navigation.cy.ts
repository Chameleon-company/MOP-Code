// Canonical navigation test suite — consolidates nav.cy.ts, inner_to_home.cy.ts,
// and signup.cy.ts into one file.
//
// Selector:
//   - Home link:     cy.get('a[href*="/en"]')   or aria-label="Go to homepage"
//   - About Us:      cy.get('a[href*="/about"]')
//   - Log In:        cy.get('a[href*="/login"]')
//   - Use Cases:     cy.get('a[href*="/usecases"]')
//   - Sign Up:       cy.get('a[href*="/signup"]')

// Shared stubs — stub the home page API calls so every visit() is backend-free
const HOME_STUBS = () => {
  cy.intercept('GET', '/api/home/categories', {
    body: { success: true, data: [{ id: 1, category_name: 'Business', description: '', cover_img: '' }] },
  }).as('categories');
  cy.intercept('GET', '/api/usecases/recent', {
    body: { success: true, data: [{ id: 1, title: 'Test Use Case', description: '', cover_img: '', tags: [] }] },
  }).as('recentUseCases');
  cy.intercept('GET', '/api/statistics/**', { body: { success: true, data: {} } }).as('statistics');
};

// 1. Global Navigation (outbound from home)
describe('Global Navigation', () => {
  beforeEach(() => {
    HOME_STUBS();
    cy.viewport(1280, 720);
    cy.visit('/');
  });

  it('should navigate to the About Us page', () => {
    // Use href selector
    cy.get('a[href*="/about"]').first().click();
    cy.url().should('include', '/about');
  });

  it('should navigate to Use Cases via Explore dropdown', () => {
    cy.get('a[href*="/usecases"]').first().click();
    cy.url().should('include', '/usecases');
  });

  it('should navigate to the Login page', () => {
    cy.get('a[href*="/login"]').first().click();
    cy.url().should('include', '/login');
  });

  it('should navigate home via the logo', () => {
    cy.visit('/about');
    cy.get('a[aria-label="Go to homepage"]').click();
    cy.url().should('match', /\/en\/?$/);
  });

  it('should navigate home via the Home nav link', () => {
    cy.visit('/about');
    // Target the nav <a> element directly by href to avoid matching any in-page "Home" text
    cy.get('nav').find('a[href*="/en"]').first().click();
    cy.url().should('match', /\/en\/?$/);
  });
});


// 2. Return Navigation to Home (from every menu-bar page)
describe('Return Navigation to Home from inner pages', () => {
  // Helper: visit a page, click the Home nav link, assert we land on /en
  const navigateHomeFrom = (path: string) => {
    cy.visit(path);
    cy.get('nav').find('a[href*="/en"]').first().click();
    cy.url().should('match', /\/en\/?$/);
  };

  it('should navigate home from About Us page', () => {
    navigateHomeFrom('/about');
  });

  it('should navigate home from Use Cases page', () => {
    HOME_STUBS();
    navigateHomeFrom('/usecases');
  });

  it('should navigate home from Blogs page', () => {
    navigateHomeFrom('/blog');
  });

  it('should navigate home from Gallery page', () => {
    navigateHomeFrom('/gallery');
  });

  it('should navigate home from Contact Us page', () => {
    navigateHomeFrom('/contact');
  });
});

// 3. Auth-Guarded Page Redirect
describe('Auth guard redirect', () => {
  it('should redirect unauthenticated users from /profile to /login', () => {
    cy.clearLocalStorage();
    cy.visit('/profile');
    cy.url({ timeout: 8000 }).should('include', '/login');
  });
});

// 4. Signup Navigation Flow
describe('Signup Navigation', () => {
  it('should navigate to the signup page via the login page', () => {
    HOME_STUBS();
    cy.visit('/');

    // Use href selector for locale-proof "Log In" click
    cy.get('a[href*="/login"]').first().click();
    cy.url().should('include', '/login');

    // On the login page, click the Sign Up link by href
    cy.get('a[href*="/signup"]').first().click();
    cy.url({ timeout: 8000 }).should('include', '/signup');
  });
});
