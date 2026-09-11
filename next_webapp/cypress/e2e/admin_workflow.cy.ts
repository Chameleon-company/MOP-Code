// General Admin E2E tests — access control, categories, contributors, blogs.
// Usecase workflows live in usecases.cy.ts. Auth is faked via localStorage;
// all API calls are stubbed with cy.intercept().

export { }; // module scope — usecases.cy.ts has a same-named seedAdmin

function seedAdmin(win: Cypress.AUTWindow) {
  win.localStorage.setItem(
    'user',
    JSON.stringify({ userId: 1, roleId: 1, roleName: 'admin', token: 'fake-token' }),
  );
}

// Contributors use a custom dropdown (button + option list), not a native
// <select> — see CustomSelect in ContributorForm.tsx.
function openDropdown(labelText: string) {
  cy.contains('label', labelText).parent().find('button').first().click();
}

// 1. Access control — the /admin layout guard: no user -> /login, wrong role
// -> /profile, admin -> renders.
describe('Admin access control', () => {
  it('redirects to /login when no user is stored', () => {
    // Visit an admin route with nothing in localStorage
    cy.visit('/admin/categories');
    cy.url({ timeout: 8000 }).should('include', '/login');
  });

  it('redirects to /login when the stored user has no token', () => {
    // Seed a user object that is missing the token field
    cy.visit('/admin/categories', {
      onBeforeLoad(win) {
        win.localStorage.setItem('user', JSON.stringify({ userId: 1, roleId: 1 }));
      },
    });
    cy.url({ timeout: 8000 }).should('include', '/login');
  });

  it('redirects a non-admin user to /profile', () => {
    cy.intercept('GET', '/api/profile', {
      statusCode: 200,
      body: { success: true, data: {} },
    }).as('getProfile');

    cy.visit('/admin/categories', {
      onBeforeLoad(win) {
        // profile/page.jsx checks its own top-level "token" key too — needs
        // seeding or it bounces this straight through to /login.
        win.localStorage.setItem('token', 'fake-token');
        win.localStorage.setItem(
          'user',
          JSON.stringify({ userId: 2, roleId: 2, roleName: 'student', token: 'fake-token' }),
        );
      },
    });

    // Should land on /profile, not bounce further to /admin
    cy.url({ timeout: 8000 }).should('include', '/profile');
    cy.url().should('not.include', '/admin');
  });

  it('renders the admin panel for an admin user', () => {
    // Stub the categories list so the page doesn't hit the real backend
    cy.intercept('GET', '/api/categories?*', {
      statusCode: 200,
      body: { success: true, data: [], pagination: { page: 1, pageSize: 10, total: 0, totalPages: 1 } },
    }).as('getCategories');

    cy.visit('/admin/categories', { onBeforeLoad: seedAdmin });

    cy.url().should('include', '/admin/categories');
    // AdminSidebar renders both a mobile and a desktop <aside> — bare
    // cy.contains('Dashboard') can match the hidden one. The page's own
    // heading is unique, so it's a cleaner signal that the panel rendered.
    cy.get('h1').contains('Categories').should('be.visible');
  });
});

// 2. Categories — list, add, edit, delete.
describe('Admin — categories list', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/categories?*', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          { id: 1, category_name: 'Data Science', description: 'DS use cases', cover_img: null },
          { id: 2, category_name: 'Web Development', description: 'WD use cases', cover_img: null },
        ],
        pagination: { page: 1, pageSize: 10, total: 2, totalPages: 1 },
      },
    }).as('getCategories');

    cy.visit('/admin/categories', { onBeforeLoad: seedAdmin });
    cy.wait('@getCategories');
  });

  it('renders the categories table', () => {
    // Scoped to <td> — ImageHoverPreview's hover caption repeats the name
    // in a hidden tooltip, which bare cy.contains() would match instead.
    cy.contains('td', 'Data Science').should('be.visible');
    cy.contains('td', 'Web Development').should('be.visible');
  });

  it('shows an empty state when there are no categories', () => {
    // Override the default stub with an empty response
    cy.intercept('GET', '/api/categories?*', {
      statusCode: 200,
      body: { success: true, data: [], pagination: { page: 1, pageSize: 10, total: 0, totalPages: 1 } },
    }).as('getEmptyCategories');

    cy.visit('/admin/categories', { onBeforeLoad: seedAdmin });
    cy.wait('@getEmptyCategories');
    // Empty-state message should replace the table
    cy.contains('No data available at the moment.').should('be.visible');
  });

  it('links to the add-category page', () => {
    cy.contains('a', 'Add New').should('have.attr', 'href').and('include', '/admin/categories/add');
  });

  it('asks for confirmation before deleting, and cancel leaves it untouched', () => {
    // Open the delete confirmation dialog
    cy.get('button[aria-label="Delete Data Science"]').click();
    cy.contains('Delete Category').should('be.visible');
    cy.contains('Are you sure you want to delete "Data Science"?').should('be.visible');

    // Cancel should close the dialog without removing the row
    cy.contains('button', 'Cancel').click();
    cy.contains('Delete Category').should('not.exist');
    cy.contains('td', 'Data Science').should('be.visible');
  });

  it('deletes a category on confirm', () => {
    cy.intercept('DELETE', '/api/categories/1', {
      statusCode: 200,
      body: { success: true },
    }).as('deleteCategory');

    // Open the dialog and confirm deletion
    cy.get('button[aria-label="Delete Data Science"]').click();
    cy.contains('button', 'Delete').click();

    cy.wait('@deleteCategory');
    cy.contains('Category deleted successfully.').should('be.visible');
  });
});

// 3. Add category
describe('Admin — add category', () => {
  beforeEach(() => {
    cy.visit('/admin/categories/add', { onBeforeLoad: seedAdmin });
  });

  it('sends the entered title and description on submit', () => {
    cy.intercept('POST', '/api/categories', {
      statusCode: 201,
      body: { success: true, data: { id: 3 } },
    }).as('createCategory');

    // Fill in the form fields
    cy.get('input[placeholder="Enter category title"]').type('Cyber Security');
    cy.get('textarea[placeholder="Enter category description"]').type('Cyber use cases');
    cy.get('button[type="submit"]').click();

    // Verify the request body contains the entered values
    cy.wait('@createCategory').its('request.body').should('deep.include', {
      category_name: 'Cyber Security',
      description: 'Cyber use cases',
      cover_img: null,
    });

    cy.contains('Category added successfully.').should('be.visible');
  });
});

// 4. Edit category
describe('Admin — edit category', () => {
  const categoryId = '1';

  beforeEach(() => {
    cy.intercept('GET', `/api/categories/${categoryId}`, {
      statusCode: 200,
      body: {
        success: true,
        data: { category_name: 'Data Science', description: 'DS use cases', cover_img: null },
      },
    }).as('getCategory');

    cy.visit(`/admin/categories/edit/${categoryId}`, { onBeforeLoad: seedAdmin });
    cy.wait('@getCategory');
  });

  it('pre-fills the form with the existing category', () => {
    // Both fields should show the values fetched from the API
    cy.get('input[placeholder="Enter category title"]').should('have.value', 'Data Science');
    cy.get('textarea[placeholder="Enter category description"]').should('have.value', 'DS use cases');
  });

  it('sends the updated fields and redirects to the list', () => {
    cy.intercept('PUT', `/api/categories/${categoryId}`, {
      statusCode: 200,
      body: { success: true },
    }).as('updateCategory');

    // The redirect lands on the categories list, which fires its own GET on
    // mount — without this it hits the real backend, gets a real 401 for
    // the fake token, and the list page's own handler bounces to /login.
    cy.intercept('GET', '/api/categories?*', {
      statusCode: 200,
      body: { success: true, data: [], pagination: { page: 1, pageSize: 10, total: 0, totalPages: 1 } },
    }).as('getCategoriesAfterUpdate');

    // Edit the title and submit
    cy.get('input[placeholder="Enter category title"]').clear().type('Data Science & AI');
    cy.get('button[type="submit"]').click();

    // Verify the PUT body contains the updated title and unchanged description
    cy.wait('@updateCategory').its('request.body').should('deep.include', {
      category_name: 'Data Science & AI',
      description: 'DS use cases',
    });

    cy.url({ timeout: 3000 }).should('include', '/admin/categories');
  });
});

// 5. Contributors — add, edit, delete, plus tests pinning down PR #2138:
// ContributorForm.tsx's hardcoded lists are stale, so project_lead and the
// newer roles/team can't be created here.
describe('Admin — add contributor', () => {
  beforeEach(() => {
    cy.visit('/admin/contributors/add', { onBeforeLoad: seedAdmin });
  });

  it('creates a student contributor with the selected team, position, and level', () => {
    cy.intercept('POST', '/api/contributors', {
      statusCode: 201,
      body: { success: true, data: { id: 1 } },
    }).as('createContributor');

    cy.get('input[placeholder="Enter full name"]').type('Ada Lovelace');
    // .clear() first — without it a retried .type() can double up the
    // value (e.g. "20262026"), tripping native max=2100 validation and
    // silently blocking submit before handleSubmit ever runs.
    cy.get('input[type="number"]').first().clear().type('2026'); // Year

    // Each selection is checked against its own trigger button before
    // moving on, so a state-update failure points at the exact field.
    openDropdown('Contributor Type');
    cy.contains('button', '👨‍🎓 Student').click();
    cy.contains('label', 'Contributor Type').parent().find('button').first().should('contain', 'Student');

    openDropdown('Team');
    cy.contains('button', 'Data Science Team').click();
    cy.contains('label', 'Team').parent().find('button').first().should('contain', 'Data Science Team');

    openDropdown('Position or Role');
    cy.contains('button', 'Data Scientist').click();
    cy.contains('label', 'Position or Role').parent().find('button').first().should('contain', 'Data Scientist');

    openDropdown('Level');
    cy.contains('button', 'Senior').click();
    cy.contains('label', 'Level').parent().find('button').first().should('contain', 'Senior');

    cy.contains('button', 'Add Contributor').click();

    // Verify the POST body contains all selected values
    cy.wait('@createContributor').its('request.body').should('deep.include', {
      name: 'Ada Lovelace',
      contributor_type: 'student',
      team: 'Data Science Team',
      position: 'Data Scientist',
      level: 'Senior',
    });

    cy.contains('Contributor added successfully.').should('be.visible');
  });

  it('does not offer "Project Lead" as a contributor type', () => {
    // Open the dropdown and verify all visible options — project_lead is a stale DB value (PR #2138)
    openDropdown('Contributor Type');
    cy.contains('button', '👨‍🎓 Student').should('be.visible');
    cy.contains('button', '👨‍🏫 Mentor').should('be.visible');
    cy.contains('button', '🏢 Company Director').should('be.visible');
    cy.contains('button', /project lead/i).should('not.exist');
  });

  it('does not offer the "Project Team" team option', () => {
    // Select Student type first — team options are dependent on contributor type
    openDropdown('Contributor Type');
    cy.contains('button', '👨‍🎓 Student').click();

    // "Project Team" is a newer addition not yet in the hardcoded list (PR #2138)
    openDropdown('Team');
    cy.contains('button', 'Data Science Team').should('be.visible');
    cy.contains('button', 'Cyber Security Team').should('be.visible');
    cy.contains('button', 'Project Team').should('not.exist');
  });

  it('does not offer the newer Data Science Team positions', () => {
    openDropdown('Contributor Type');
    cy.contains('button', '👨‍🎓 Student').click();

    openDropdown('Team');
    cy.contains('button', 'Data Science Team').click();

    // Check which positions are and aren't offered (PR #2138 gap)
    openDropdown('Position or Role');
    cy.contains('button', 'Data Scientist').should('be.visible');
    cy.contains('button', 'Data Science Team Lead').should('be.visible');
    cy.contains('button', 'Data Science Assistant Team Lead').should('not.exist');
    cy.contains('button', 'Project Lead').should('not.exist');
  });
});

// 6. Edit contributor
describe('Admin — edit contributor', () => {
  const contributorId = '42';

  it('pre-fills a student contributor correctly', () => {
    cy.intercept('GET', `/api/contributors/${contributorId}`, {
      statusCode: 200,
      body: {
        success: true,
        data: {
          id: contributorId,
          name: 'Ada Lovelace',
          year: 2026,
          trimester: 1,
          contributor_type: 'student',
          team: 'Data Science Team',
          position: 'Data Scientist',
          level: 'Senior',
          display_order: 0,
          is_active: true,
        },
      },
    }).as('getContributor');

    cy.visit(`/admin/contributors/edit/${contributorId}`, { onBeforeLoad: seedAdmin });
    cy.wait('@getContributor');

    // Name field and selected position should reflect the fetched data
    cy.get('input[placeholder="Enter full name"]').should('have.value', 'Ada Lovelace');
    cy.contains('button', 'Data Scientist').should('be.visible');
  });

  it('shows no contributor type selected for a project_lead record (the #2138 gap)', () => {
    cy.intercept('GET', `/api/contributors/${contributorId}`, {
      statusCode: 200,
      body: {
        success: true,
        data: {
          id: contributorId,
          name: 'Grace Hopper',
          year: 2026,
          trimester: 1,
          contributor_type: 'project_lead',
          team: null,
          position: null,
          level: null,
          display_order: 0,
          is_active: true,
        },
      },
    }).as('getContributor');

    cy.visit(`/admin/contributors/edit/${contributorId}`, { onBeforeLoad: seedAdmin });
    cy.wait('@getContributor');

    // "project_lead" isn't a form option, so it falls back to the placeholder.
    cy.contains('label', 'Contributor Type').parent().contains('button', 'Select type').should('be.visible');
  });
});

// 7. Delete contributor
describe('Admin — contributors list: delete', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/contributors', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          { id: 1, name: 'Ada Lovelace', year: 2026, trimester: 1, contributor_type: 'student', team: 'Data Science Team', position: 'Data Scientist', level: 'Senior', is_active: true },
        ],
      },
    }).as('getContributors');

    cy.visit('/admin/contributors', { onBeforeLoad: seedAdmin });
    cy.wait('@getContributors');
  });

  it('asks for confirmation before deleting, and cancel leaves it in place', () => {
    // Open the delete confirmation dialog
    cy.get('button[aria-label="Delete Ada Lovelace"]').click();
    cy.contains('Delete Contributor').should('be.visible');
    cy.contains('Are you sure you want to delete').should('be.visible');
    cy.contains('Ada Lovelace').should('be.visible');

    // Cancel should dismiss the dialog without removing the row
    cy.contains('button', 'Cancel').click();
    cy.contains('Delete Contributor').should('not.exist');
  });

  it('deletes a contributor on confirm', () => {
    cy.intercept('DELETE', '/api/contributors/1', {
      statusCode: 200,
      body: { success: true },
    }).as('deleteContributor');

    // Open the dialog and confirm deletion
    cy.get('button[aria-label="Delete Ada Lovelace"]').click();
    cy.contains('button', 'Delete').click();

    cy.wait('@deleteContributor');
    cy.contains('Contributor deleted successfully.').should('be.visible');
  });
});

// 8. Blogs — list, add, edit, delete. Requests are FormData, so tests check
// behavior (toast, redirect) rather than the body. Content is CKEditor5,
// loaded dynamically — `.ck-editor__editable` is its real editable class.
describe('Admin — blogs list', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/blogs?*', {
      statusCode: 200,
      body: {
        success: true,
        data: [
          { id: 1, title: 'MOP Launches Real-Time Transit Layer', description: 'New feature', published_date: '2026-08-01' },
          { id: 2, title: 'Behind the Scenes: The GridFS Migration', description: 'Engineering deep dive', published_date: '2026-08-15' },
        ],
        pagination: { page: 1, pageSize: 10, total: 2, totalPages: 1 },
      },
    }).as('getBlogs');

    cy.visit('/admin/blogs', { onBeforeLoad: seedAdmin });
    cy.wait('@getBlogs');
  });

  it('renders the blog rows', () => {
    // Scoped to <td> — ImageHoverPreview's alt text repeats the title in a
    // hidden hover caption, which bare cy.contains() would match instead.
    cy.contains('td', 'MOP Launches Real-Time Transit Layer').should('be.visible');
    cy.contains('td', 'Behind the Scenes: The GridFS Migration').should('be.visible');
  });

  it('asks for confirmation before deleting, and cancel leaves it in place', () => {
    // Open the delete dialog via the second button (Delete) in the row
    cy.contains('tr', 'MOP Launches Real-Time Transit Layer').within(() => {
      cy.get('button').eq(1).click();
    });

    cy.contains('Delete Blog').should('be.visible');
    // Cancel should dismiss the dialog without removing the row
    cy.contains('button', 'Cancel').click();

    cy.contains('Delete Blog').should('not.exist');
    cy.contains('td', 'MOP Launches Real-Time Transit Layer').should('be.visible');
  });

  it('deletes a blog on confirm', () => {
    cy.intercept('DELETE', '/api/blogs/1', {
      statusCode: 200,
      body: { success: true },
    }).as('deleteBlog');

    // Open the dialog and confirm deletion
    cy.contains('tr', 'MOP Launches Real-Time Transit Layer').within(() => {
      cy.get('button').eq(1).click();
    });
    cy.contains('button', 'Delete').click();

    cy.wait('@deleteBlog');
    cy.contains('Blog deleted successfully.').should('be.visible');
  });
});

// 9. Add blog
describe('Admin — add blog', () => {
  beforeEach(() => {
    cy.visit('/admin/blogs/add', { onBeforeLoad: seedAdmin });
  });

  it('shows validation errors when submitted empty', () => {
    // Submit without filling any fields — all required errors should appear
    cy.contains('button', 'Save Blog').click();

    cy.contains('Title is required').should('be.visible');
    cy.contains('Description is required').should('be.visible');
    cy.contains('Date is required').should('be.visible');
    cy.contains('Cover image is required').should('be.visible');
    cy.contains('Content is required').should('be.visible');
  });

  it('creates a blog post and redirects to the list', () => {
    cy.intercept('POST', '/api/blogs', {
      statusCode: 201,
      body: { success: true, data: { id: 3 } },
    }).as('createBlog');

    // Fill in all required fields
    cy.get('input[placeholder="Enter blog title"]').type('Cypress Coverage for the Admin Panel');
    cy.get('input[placeholder="Short description or subtitle"]').type('Closing the e2e gaps');
    cy.get('input[type="date"]').type('2026-09-09');
    // CKEditor adds a hidden file input on mount — scope to the label to stay unique.
    cy.contains('label', 'Cover Image').parent().find('input[type="file"]')
      .selectFile('cypress/fixtures/cover.png', { force: true });

    // Synthetic keystrokes outpace CKEditor's engine — set content via its API instead.
    cy.get('.ck-editor__editable', { timeout: 15000 }).then(($el) => {
      const editor = ($el[0] as any).ckeditorInstance;
      editor.setData('<p>This post covers the new admin test suites.</p>');
    });

    cy.contains('button', 'Save Blog').click();

    cy.wait('@createBlog');
    cy.contains('Blog added successfully.').should('be.visible');
    cy.url({ timeout: 3000 }).should('include', '/admin/blogs');
  });
});

// 10. Edit blog
describe('Admin — edit blog', () => {
  const blogId = '7';

  beforeEach(() => {
    cy.intercept('GET', `/api/blogs/${blogId}`, {
      statusCode: 200,
      body: {
        success: true,
        data: {
          title: 'Original Title',
          description: 'Original description',
          published_date: '2026-08-01',
          content: '<p>Existing content</p>',
          cover_img: 'https://cdn.example.com/blogs/original.png',
        },
      },
    }).as('getBlog');

    cy.visit(`/admin/blogs/edit/${blogId}`, { onBeforeLoad: seedAdmin });
    cy.wait('@getBlog');
  });

  it('pre-fills the form with the existing blog', () => {
    // Both fields should show the values fetched from the API
    cy.get('input[placeholder="Enter blog title"]').should('have.value', 'Original Title');
    cy.get('input[placeholder="Short description or subtitle"]').should('have.value', 'Original description');
  });

  it('updates the title and redirects to the list', () => {
    cy.intercept('PUT', `/api/blogs/${blogId}`, {
      statusCode: 200,
      body: { success: true },
    }).as('updateBlog');

    // Edit the title and submit
    cy.get('input[placeholder="Enter blog title"]').clear().type('Updated Title');
    cy.contains('button', 'Save Blog').click();

    cy.wait('@updateBlog');
    cy.url({ timeout: 3000 }).should('include', '/admin/blogs');
  });
});
