export {};

describe('F2 — edit use case category IDs', () => {
  const id = '660f1f77bcf86cd799439099';
  const first = '507f1f77bcf86cd799439011';
  const second = '507f1f77bcf86cd799439012';

  beforeEach(() => {
    cy.intercept('GET', `/api/usecases/${id}`, {
      success: true,
      data: {
        id, title: 'Category regression', description: 'Test description',
        cover_img: null, category: { id: first }, tags: [],
      },
    });
    cy.intercept('GET', '/api/categories', {
      success: true,
      data: [{ id: first, name: 'First category' }, { id: second, name: 'Second category' }],
    });
    cy.intercept('PUT', `/api/usecases/${id}`, {
      statusCode: 400,
      body: { success: false, message: 'Test stops after checking the request' },
    }).as('save');
    cy.visit(`/admin/use-cases/edit/${id}`, {
      onBeforeLoad(win) {
        win.localStorage.setItem('user', JSON.stringify({
          userId: 1, roleId: 1, roleName: 'admin', token: 'fake-token',
        }));
      },
    });
    cy.get('select').should('have.value', first);
  });

  it('preserves the category string when only the title changes', () => {
    cy.get('input[placeholder="Enter use case title"]').clear().type('Changed title');
    cy.get('button[type="submit"]').click();
    cy.wait('@save').then(({ request }) => {
      expect(request.body.title).to.eq('Changed title');
      expect(request.body.category_id).to.eq(first);
    });
  });

  it('sends a newly selected category as a string', () => {
    cy.get('select').select(second);
    cy.get('button[type="submit"]').click();
    cy.wait('@save').its('request.body.category_id').should('eq', second);
  });

  it('sends null when the category is cleared', () => {
    cy.get('select').select('');
    cy.get('button[type="submit"]').click();
    cy.wait('@save').its('request.body.category_id').should('eq', null);
  });
});
