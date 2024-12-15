// cypress/integration/navigation.spec.js

describe('Navigation', () => {
  it('should navigate to the sign-up page', () => {
    // Visit the home page
    cy.visit('http://localhost:3000/');

    // Find and click the link to the Sign Up page
    cy.get('a[href="/signup"]').click();

    // Wait for the sign-up page to load and check if the URL includes '/signup'
    cy.url().should('include', '/signup');

  });
});
