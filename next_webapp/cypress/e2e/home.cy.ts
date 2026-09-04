// Home page E2E tests — verifies key content loads and critical interactions work
// API calls are stubbed so tests run without a live backend.
//
// Selectors:
//   - Page heading:    cy.get('h1')
//   - Header nav:      cy.get('nav[aria-label="Main navigation"]')
//   - Category card:   cy.contains('<name>') — cards are plain <div>s, not <a> links
//   - Chatbot button:  cy.get('button[aria-label="Open chat"]')
export {};

// Shared stubs — stub home page API calls so every visit() is backend-free
const CATEGORIES_STUB = {
  success: true,
  data: [
    { id: 1, category_name: 'Business and Economy', description: 'Test category', cover_img: '' },
    { id: 2, category_name: 'Transport', description: 'Test category 2', cover_img: '' },
  ],
};

const USECASES_STUB = {
  success: true,
  data: [
    { id: 1, title: 'Test Use Case', description: 'Test description', cover_img: '', tags: [] },
    { id: 2, title: 'Another Use Case', description: 'Another description', cover_img: '', tags: [] },
  ],
};

// 1. Home Page
describe('Home Page', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/home/categories', { body: CATEGORIES_STUB }).as('categories');
    cy.intercept('GET', '/api/usecases/recent', { body: USECASES_STUB }).as('recentUseCases');
    cy.visit('/');
  });

  it('renders the page heading', () => {
    // The page should contain a visible h1 heading
    cy.get('h1').should('be.visible');
  });

  it('renders the header navigation', () => {
    // Header element should be visible
    cy.get('header').should('be.visible');
    // Main navigation should be present
    cy.get('nav[aria-label="Main navigation"]').should('be.visible');
    // About Us link should be in the nav
    cy.contains('About Us').should('be.visible');
  });

  it('loads and displays category cards from API', () => {
    // Wait for the categories API response before asserting
    cy.wait('@categories');
    // Stubbed category name should be visible on the page
    cy.contains('Business and Economy').should('be.visible');
  });

  it('loads and displays recent use case cards from API', () => {
    // Wait for the recent use cases API response before asserting
    cy.wait('@recentUseCases');
    // Stubbed use case title should be visible on the page
    cy.contains('Test Use Case').should('be.visible');
  });

  it('displays all category cards returned by the API', () => {
    // Wait for categories to load before asserting
    cy.wait('@categories');
    // Both stubbed categories should be visible as rendered cards
    cy.contains('Business and Economy').should('be.visible');
    cy.contains('Transport').should('be.visible');
  });

  it('the chatbot toggle button is visible on the home page', () => {
    // Wait for both API responses — chatbot may only mount after data is ready
    cy.wait(['@categories', '@recentUseCases']);
    cy.get('button[aria-label="Open chat"]').should('be.visible');
  });
});
