import { defineConfig } from 'cypress';

export default defineConfig({
  allowCypressEnv: false,

  // API routes do NOT use the locale prefix
  env: {
    apiBaseUrl: 'http://localhost:3000',
  },

  e2e: {
    // Add locale /en for default testing
    // Set the baseUrl to the running Next.js application
    baseUrl: 'http://localhost:3000/en',  // Ensure your app runs here during testing
    // include e2e and integration test files
    specPattern: 'cypress/{e2e,integration}/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/e2e.ts',
    viewportWidth: 1280,
    viewportHeight: 720,
    setupNodeEvents(on, config) {
      return config;
    },
  },
});