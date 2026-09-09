/// <reference types="cypress" />

// Overwrite cy.request() so that paths starting with /api/ use the
// apiBaseUrl (without locale prefix) instead of the default baseUrl (/en).
Cypress.Commands.overwrite('request', (originalFn, ...args) => {
    // Derive apiBase from baseUrl to avoid cy.env() nesting errors and Cypress.env() security warnings
    const baseUrl = Cypress.config('baseUrl');
    const apiBase = typeof baseUrl === 'string' ? baseUrl.replace(/\/en$/, '') : 'http://localhost:3000';
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