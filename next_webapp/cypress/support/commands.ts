/// <reference types="cypress" />
// ***********************************************
// This example commands.ts shows you how to
// create various custom commands and overwrite
// existing commands.
//
// For more comprehensive examples of custom
// commands please read more here:
// https://on.cypress.io/custom-commands
// ***********************************************
//
//
// -- This is a parent command --
// Cypress.Commands.add('login', (email, password) => { ... })
//
//
// -- This is a child command --
// Cypress.Commands.add('drag', { prevSubject: 'element'}, (subject, options) => { ... })
//
//
// -- This is a dual command --
// Cypress.Commands.add('dismiss', { prevSubject: 'optional'}, (subject, options) => { ... })
//
//
// -- This will overwrite an existing command --
// Cypress.Commands.overwrite('visit', (originalFn, url, options) => { ... })
//
// declare global {
//   namespace Cypress {
//     interface Chainable {
//       login(email: string, password: string): Chainable<void>
//       drag(subject: string, options?: Partial<TypeOptions>): Chainable<Element>
//       dismiss(subject: string, options?: Partial<TypeOptions>): Chainable<Element>
//       visit(originalFn: CommandOriginalFn, url: string, options: Partial<VisitOptions>): Chainable<Element>
//     }
//   }
// }

// Overwrite cy.request() so that paths starting with /api/ use the
// apiBaseUrl (without locale prefix) instead of the default baseUrl
// which includes /en.  This keeps API tests working without needing
// to modify individual test files.
Cypress.Commands.overwrite('request', (originalFn, ...args) => {
    const apiBase = Cypress.env('apiBaseUrl');
    if (!apiBase) return originalFn(...args);

    // cy.request(url), cy.request(method, url), or cy.request(options)
    if (typeof args[0] === 'string' && args.length === 1 && args[0].startsWith('/api/')) {
        // cy.request('/api/...')
        args[0] = `${apiBase}${args[0]}`;
    } else if (typeof args[0] === 'string' && typeof args[1] === 'string' && args[1].startsWith('/api/')) {
        // cy.request('GET', '/api/...')
        args[1] = `${apiBase}${args[1]}`;
    } else if (typeof args[0] === 'object' && args[0].url && args[0].url.startsWith('/api/')) {
        // cy.request({ url: '/api/...' })
        args[0] = { ...args[0], url: `${apiBase}${args[0].url}` };
    }

    return originalFn(...args);
});