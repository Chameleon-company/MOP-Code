// Admin Use Cases E2E tests — add (notebook upload), edit (notebook
// tri-state), and delete workflows for the use-cases admin section.
// Auth is faked via localStorage; all API calls are stubbed with cy.intercept().
//
// Selectors:
//   - Notebook input:    cy.get('input[type="file"][accept*="ipynb"]')
//   - Title input:       cy.get('input[placeholder="Enter use case title"]')
//   - Submit button:     cy.get('button[type="submit"]')
// List rows have no aria-label on their action buttons, so rows are scoped
// by title text and the delete button is the row's 2nd button (1st is Edit).

export {};

function seedAdmin(win: Cypress.AUTWindow) {
  win.localStorage.setItem(
    'user',
    JSON.stringify({ userId: 1, roleId: 1, roleName: 'admin', token: 'fake-token' }),
  );
}

describe('Admin — add use case: notebook upload', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/categories', {
      statusCode: 200,
      body: { success: true, data: [] },
    }).as('getCategories');

    cy.visit('/admin/use-cases/add', { onBeforeLoad: seedAdmin });
    cy.wait('@getCategories');
  });

  // Verifies the initial state of the upload zone before any file is chosen.
  it('shows the upload prompt with no file selected', () => {
    cy.contains('Upload Python notebook').should('be.visible');
  });

  // Verifies that a non-.ipynb file is rejected and the upload zone stays unchanged.
  it('rejects a file that is not .ipynb', () => {
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/not-a-notebook.txt', { force: true });

    cy.contains('Please upload a valid .ipynb file.').should('be.visible');
    // Nothing was accepted, so the upload zone still shows the placeholder.
    cy.contains('Upload Python notebook').should('be.visible');
  });

  // Verifies that a .ipynb file with invalid JSON is caught and the upload zone stays unchanged.
  it('rejects a .ipynb file that is not valid notebook JSON', () => {
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/malformed.ipynb', { force: true });

    cy.contains('Invalid notebook file').should('be.visible');
    cy.contains('Upload Python notebook').should('be.visible');
  });

  // Verifies that a valid notebook is accepted and its filename and removal option are shown.
  it('accepts a valid notebook and shows it selected', () => {
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/sample.ipynb', { force: true });

    cy.contains('sample.ipynb').should('be.visible');
    cy.contains('Remove notebook').should('be.visible');
  });

  // Verifies that clicking "Remove notebook" clears the selection and restores the upload prompt.
  it('clears the selection when "Remove notebook" is clicked', () => {
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/sample.ipynb', { force: true });
    cy.contains('sample.ipynb').should('be.visible');

    cy.contains('Remove notebook').click();

    cy.contains('Upload Python notebook').should('be.visible');
    cy.contains('Remove notebook').should('not.exist');
  });

  // Verifies that the notebook JSON is sent as the content field and the page redirects on success.
  it('sends the notebook JSON as the content field and redirects on success', () => {
    cy.intercept('POST', '/api/usecases', {
      statusCode: 201,
      body: { success: true, data: { id: 'cypress-test-id' } },
    }).as('createUseCase');

    cy.get('input[placeholder="Enter use case title"]').type('Cypress notebook upload test');
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/sample.ipynb', { force: true });
    cy.contains('sample.ipynb').should('be.visible');

    cy.get('button[type="submit"]').click();

    cy.wait('@createUseCase').then((interception) => {
      const body = interception.request.body;
      expect(body.title).to.eq('Cypress notebook upload test');
      expect(body.content).to.be.a('string');

      const notebook = JSON.parse(body.content);
      expect(notebook.cells).to.be.an('array');
    });

    cy.contains('Use case added successfully.').should('be.visible');
    cy.url({ timeout: 3000 }).should('include', '/admin/use-cases');
  });
});

describe('Admin — edit use case: notebook flow', () => {
  const useCaseId = '660f1f77bcf86cd799439099';

  beforeEach(() => {
    cy.intercept('GET', `/api/usecases/${useCaseId}`, {
      statusCode: 200,
      body: {
        success: true,
        data: {
          id: useCaseId,
          title: 'Existing Use Case',
          description: 'Existing description',
          cover_img: null,
          category: null,
          content_file_id: 'gridfs-file-id-123',
          tags: [{ name: 'existing-tag' }],
        },
      },
    }).as('getUseCase');

    cy.intercept('GET', '/api/categories', {
      statusCode: 200,
      body: { success: true, data: [] },
    }).as('getCategories');

    cy.visit(`/admin/use-cases/edit/${useCaseId}`, { onBeforeLoad: seedAdmin });
    cy.wait(['@getUseCase', '@getCategories']);
  });

  // Verifies that an already-uploaded notebook is shown without requiring re-upload.
  it('shows the existing notebook without re-uploading', () => {
    cy.contains('Notebook already uploaded').should('be.visible');
    cy.contains('Remove notebook').should('be.visible');
  });

  // Verifies that submitting without touching the notebook omits the content field from the PUT body.
  it('omits content from the PUT body when the notebook is left untouched', () => {
    cy.intercept('PUT', `/api/usecases/${useCaseId}`, { statusCode: 200, body: { success: true, data: {} } }).as('updateUseCase');

    cy.get('button[type="submit"]').click();

    cy.wait('@updateUseCase').its('request.body').should('not.have.property', 'content');
  });

  // Verifies that removing the existing notebook sends content: null in the PUT body.
  it('sends content: null when the existing notebook is removed', () => {
    cy.intercept('PUT', `/api/usecases/${useCaseId}`, { statusCode: 200, body: { success: true, data: {} } }).as('updateUseCase');

    cy.contains('Remove notebook').click();
    cy.contains('Upload Python notebook').should('be.visible');

    cy.get('button[type="submit"]').click();

    cy.wait('@updateUseCase').its('request.body.content').should('be.null');
  });

  // Verifies that uploading a malformed replacement is rejected and the existing notebook state is preserved.
  it('rejects a malformed replacement notebook', () => {
    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/malformed.ipynb', { force: true });

    cy.contains('Invalid notebook file').should('be.visible');
    // Rejected file doesn't count as a replacement or a removal.
    cy.contains('Notebook already uploaded').should('be.visible');
  });

  // Verifies that uploading a replacement notebook sends the new notebook JSON in the PUT body.
  it('sends the new notebook JSON when a replacement is uploaded', () => {
    cy.intercept('PUT', `/api/usecases/${useCaseId}`, { statusCode: 200, body: { success: true, data: {} } }).as('updateUseCase');

    cy.get('input[type="file"][accept*="ipynb"]')
      .selectFile('cypress/fixtures/sample.ipynb', { force: true });
    cy.contains('sample.ipynb').should('be.visible');

    cy.get('button[type="submit"]').click();

    cy.wait('@updateUseCase').then((interception) => {
      const body = interception.request.body;
      expect(body.content).to.be.a('string');
      expect(JSON.parse(body.content).cells).to.be.an('array');
    });
  });

  // Verifies that a successful update redirects back to the use case list.
  it('redirects to the use case list after a successful update', () => {
    cy.intercept('PUT', `/api/usecases/${useCaseId}`, { statusCode: 200, body: { success: true, data: {} } }).as('updateUseCase');

    cy.get('button[type="submit"]').click();
    cy.wait('@updateUseCase');

    cy.url({ timeout: 3000 }).should('include', '/admin/use-cases');
  });
});

describe('Admin — use cases: delete', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/categories?*', {
      statusCode: 200,
      body: { success: true, data: [{ id: 1, category_name: 'Data Science' }] },
    }).as('getCategories');

    cy.intercept('GET', '/api/usecases?*', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          {
            id: 101,
            title: 'EV Charging Demand',
            // Real shape from toUseCaseDTO — no flat category_id field.
            category: { id: '6a63080e16bb69198c5b835d', legacy_id: '1', category_name: 'Data Science' },
            description: 'Forecasting demand',
            cover_img: null,
          },
          {
            id: 102,
            title: 'Bike Share Utilisation',
            category: { id: '6a63080e16bb69198c5b835e', legacy_id: '1', category_name: 'Data Science' },
            description: 'Usage patterns',
            cover_img: null,
          },
        ],
        pagination: { page: 1, pageSize: 10, total: 2, totalPages: 1 },
      },
    }).as('getUseCases');

    cy.visit('/admin/use-cases', { onBeforeLoad: seedAdmin });
    cy.wait(['@getCategories', '@getUseCases']);
  });

  // Verifies that use case rows are rendered and the category name is sourced from the embedded category ref.
  it('renders the use case rows with their category name', () => {
    // Scoped to <td> — ImageHoverPreview's alt text repeats the title in a
    // hidden hover caption, which bare cy.contains() would match instead.
    cy.contains('td', 'EV Charging Demand').should('be.visible');
    cy.contains('td', 'Bike Share Utilisation').should('be.visible');

    // Regression check: category_name comes straight off the embedded
    // category ref, not a join against /api/categories by a category_id
    // field the Mongo doc doesn't have.
    cy.contains('tr', 'EV Charging Demand').should('contain', 'Data Science');
  });

  // Verifies that a delete confirmation dialog appears and cancelling leaves the row intact.
  it('asks for confirmation before deleting, and cancel leaves the row in place', () => {
    cy.contains('tr', 'EV Charging Demand').within(() => {
      cy.get('button').eq(1).click();
    });

    cy.contains('Delete Use Case').should('be.visible');
    cy.contains('Are you sure you want to delete "EV Charging Demand"?').should('be.visible');

    cy.contains('button', 'Cancel').click();
    cy.contains('Delete Use Case').should('not.exist');
    cy.contains('td', 'EV Charging Demand').should('be.visible');
  });

  // Verifies that confirming delete calls the API, shows a success toast, and refetches the list.
  it('deletes a use case on confirm and refetches the list', () => {
    cy.intercept('DELETE', '/api/usecases/101', {
      statusCode: 200,
      body: { success: true },
    }).as('deleteUseCase');

    cy.intercept('GET', '/api/usecases?*', {
      statusCode: 200,
      body: {
        success: true,
        data: [{ id: 102, title: 'Bike Share Utilisation', category: { id: '6a63080e16bb69198c5b835e', legacy_id: '1', category_name: 'Data Science' }, description: 'Usage patterns', cover_img: null }],
        pagination: { page: 1, pageSize: 10, total: 1, totalPages: 1 },
      },
    }).as('getUseCasesAfterDelete');

    cy.contains('tr', 'EV Charging Demand').within(() => {
      cy.get('button').eq(1).click();
    });
    cy.contains('button', 'Delete').click();

    cy.wait('@deleteUseCase');
    cy.contains('Use case deleted successfully.').should('be.visible');

    cy.wait('@getUseCasesAfterDelete');
    cy.contains('EV Charging Demand').should('not.exist');
    cy.contains('td', 'Bike Share Utilisation').should('be.visible');
  });
});
