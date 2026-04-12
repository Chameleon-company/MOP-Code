// cypress/integration/navigation.spec.js

describe('Navigation', () => {
  it('should navigate to the about us page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the About Us page
    cy.contains('About Us').click(); // Assuming the link text is 'About Us'

    // Wait for the about us page to load and check if the URL includes '/about'
    cy.url().should('include', '/about');

  });

  it('should navigate to the use cases page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the Use Cases page by its class
    cy.get('.nav-link').contains('Use Cases').click();

    // Wait for the use cases page to load and check if the URL includes '/UseCases'
    cy.url().should('include', '/UseCases');

  });

  it('should navigate to the statistics page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the Statistics page
    cy.get('a[href="/statistics"]').click();

    // Wait for the statistics page to load and check if the URL includes '/statistics'
    cy.url().should('include', '/statistics');

  });

  it('should navigate to the upload page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the Upload page
    cy.get('a[href="/upload"]').click();

    // Wait for the upload page to load and check if the URL includes '/upload'
    cy.url().should('include', '/upload');

  });

  it('should navigate to the login page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the Sign Up page
    cy.get('a[href="/login"]').click();

    // Wait for the sign-up page to load and check if the URL includes '/signup'
    cy.url().should('include', '/login');
  });

  it('should navigate to the home page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the home page in the navigation bar
    cy.contains('Home').click(); // Assuming the link text is 'Home'

    // Wait for the home page to load and check if the URL is the base URL of the application (i.e., the home page)
    cy.url().should('eq', 'http://localhost:3000/');
  });


});
