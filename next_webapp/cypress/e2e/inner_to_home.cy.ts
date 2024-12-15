describe('Navigation', () => {
  it('should navigate to the home page from the about us page', () => {
    // Visit the about us page
    cy.visit('http://localhost:3000/about');

    // Find and click the link to the home page
    cy.contains('Home').click(); // Assuming the link text is 'Home'

    // Wait for the home page to load and check if the URL is the base URL of the application (i.e., the home page)
    cy.url().should('eq', 'http://localhost:3000/');

  });

  it('should navigate to the home page from the use cases page', () => {
    // Visit the use cases page
    cy.visit('http://localhost:3000/UseCases');

    // Find and click the link to the home page
    cy.contains('Home').click(); // Assuming the link text is 'Home'

    // Wait for the home page to load and check if the URL is the base URL of the application (i.e., the home page)
    cy.url().should('eq', 'http://localhost:3000/');

  });

  it('should navigate to the home page from the statistics page', () => {
    // Visit the use cases page
    cy.visit('http://localhost:3000/statistics');

    // Find and click the link to the home page
    cy.contains('Home').click(); // Assuming the link text is 'Home'

    // Wait for the home page to load and check if the URL is the base URL of the application (i.e., the home page)
    cy.url().should('eq', 'http://localhost:3000/');

  });



});