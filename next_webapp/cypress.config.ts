import { defineConfig } from "cypress";

export default defineConfig({
  e2e: {
    // Set the baseUrl to the running Next.js application
    baseUrl: 'http://localhost:3000',  // Ensure your app runs here during testing
    
    // Optional: specify the pattern to look for test files
    specPattern: 'cypress/integration/**/*.cy.{js,jsx,ts,tsx}',
    
    // Optional: set the viewport size
    viewportWidth: 1280,
    viewportHeight: 720,
    
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
  },
});
