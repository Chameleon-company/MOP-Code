export {};

describe('F1 - add use case category IDs', () => {
  const categoryId = '507f1f77bcf86cd799439011';

  beforeEach(() => {
    cy.intercept('GET', '/api/categories', {
      success: true,
      data: [
        {
          id: categoryId,
          category_name: 'Test Category',
        },
      ],
    });

    cy.intercept('POST', '/api/usecases', {
      statusCode: 400,
      body: {
        success: false,
        message: 'Test stops after checking the request',
      },
    }).as('save');

    cy.visit('/admin/use-cases/add', {
      onBeforeLoad(win) {
        win.localStorage.setItem('user', JSON.stringify({
          userId: 1,
          roleId: 1,
          roleName: 'admin',
          token: 'fake-token',
        }));
      },
    });

    cy.get('input[placeholder="Enter use case title"]')
      .type('Test Use Case');
  });

  it('sends the selected category ID as a string', () => {
    cy.get('form select').select(categoryId);
    cy.get('form button[type="submit"]').click();

    cy.wait('@save')
      .its('request.body.category_id')
      .should('eq', categoryId);
  });

  it('sends null when no category is selected', () => {
    cy.get('form button[type="submit"]').click();

    cy.wait('@save')
      .its('request.body.category_id')
      .should('eq', null);
  });

  it('sends null when the selected category is cleared', () => {
    cy.get('form select').select(categoryId);
    cy.get('form select').select('');
    cy.get('form button[type="submit"]').click();

    cy.wait('@save')
      .its('request.body.category_id')
      .should('eq', null);
  });
});
