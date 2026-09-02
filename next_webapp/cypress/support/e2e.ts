import './commands';

// Suppress React hydration errors globally — universal SSR noise in Next.js dev mode.
// NEXT_NOT_FOUND, NEXT_REDIRECT, and ResizeObserver are NOT suppressed here; scope those locally.
Cypress.on('uncaught:exception', (err) => {
    const hydrationErrors = [
        // React hydration mismatches (common in Next.js SSR + dev mode)
        /hydrat/i,
        // Minified React error codes for hydration failures
        /Minified React error #418/,
        /Minified React error #423/,
        /Minified React error #425/,
    ];
    if (hydrationErrors.some((pattern) => pattern.test(err.message))) {
        return false;
    }
});