// ***********************************************************
// This example support/e2e.ts is processed and
// loaded automatically before your test files.
//
// This is a great place to put global configuration and
// behavior that modifies Cypress.
//
// You can change the location of this file or turn off
// automatically serving support files with the
// 'supportFile' configuration option.
//
// You can read more here:
// https://on.cypress.io/configuration
// ***********************************************************

// Import commands.js using ES2015 syntax:
import './commands'

// Alternatively you can use CommonJS syntax:
// require('./commands')

// Suppress app-side exceptions that are unrelated to the test assertions.
// Returning false prevents Cypress from failing the test on these known-noisy errors.
Cypress.on('uncaught:exception', (err) => {
    const knownNoise = [
        // React hydration mismatches (common in Next.js SSR + dev mode)
        /hydrat/i,
        // Minified React error codes for hydration failures
        /Minified React error #418/,
        /Minified React error #423/,
        /Minified React error #425/,
        // Next.js router / navigation internal errors
        /NEXT_NOT_FOUND/,
        /NEXT_REDIRECT/,
        // ResizeObserver loop warnings from third-party widgets
        /ResizeObserver loop/,
    ];
    if (knownNoise.some((pattern) => pattern.test(err.message))) {
        return false;
    }
})