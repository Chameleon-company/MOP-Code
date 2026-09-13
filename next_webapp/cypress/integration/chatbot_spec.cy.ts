// Chatbot tests — selectors match actual chatbot.tsx component
// Toggle: button[aria-label="Open chat"]
// Chat window: .chat-window
// Input: input[placeholder="Type a message..."]
// Send: button[aria-label="Send message"]
// Messages: .message (individual bubbles)
//
// NLP navigate_log_in triggers: "log in", "sign in", "access account", "open login"
import enMessages from '../../src/app/chatbot/en.json';

describe('Chatbot Functionality Tests', () => {
  beforeEach(() => {
    // Stub the search API to return empty results (prevents 500 crashing the chatbot)
    cy.intercept('POST', '/api/search-use-cases', { body: { filteredStudies: [] } }).as('searchAPI');
    // Visit homepage to reach chatbot
    cy.visit('/');
  });

  it('should open the chatbot when the toggle button is clicked', () => {
    // Check that the chatbot is closed initially
    cy.get('.chat-window').should('not.exist');

    // Click the chatbot toggle button
    cy.get('button[aria-label="Open chat"]').click();

    // Assert that the chatbot window is visible
    cy.get('.chat-window').should('be.visible');
  });

  it('should close the chatbot when clicked again', () => {
    // Click the chatbot toggle button to open it
    cy.get('button[aria-label="Open chat"]').click();

    // Check that the chatbot is visible
    cy.get('.chat-window').should('be.visible');

    // Click the chatbot toggle button to close it
    cy.get('button[aria-label="Close chat"]').first().click();

    // Assert that the chatbot window is closed
    cy.get('.chat-window').should('not.exist');
  });

  it('should send a message and display it in the chat', () => {
    // Click the chatbot toggle button to open it
    cy.get('button[aria-label="Open chat"]').click();

    // Type a message in the input field
    cy.get('input[placeholder="Type a message..."]').type('Hello');

    // Click the send button
    cy.get('button[aria-label="Send message"]').click();

    // Check that the user's message is displayed in the chat window
    cy.get('.message').should('contain', 'Hello');
  });

  it('should display a bot fallback response for unrecognized input', () => {
    // Click the chatbot toggle button to open it
    cy.get('button[aria-label="Open chat"]').click();

    // Type an unrecognized message
    cy.get('input[placeholder="Type a message..."]').type('xyzunknowninput123');

    // Click the send button
    cy.get('button[aria-label="Send message"]').click();

    // Wait for API call, then verify it contains the fallback message
    cy.wait('@searchAPI');
    cy.get('.message', { timeout: 8000 }).should('contain', enMessages.fallback.response);
  });

  it('should navigate to login page when user types a recognized login command', () => {
    // Click the chatbot toggle button to open it
    cy.get('button[aria-label="Open chat"]').click();

    // Type a recognized command, e.g. "open login"
    cy.get('input[placeholder="Type a message..."]').type('open login');

    // Click the send button
    cy.get('button[aria-label="Send message"]').click();

    // The bot send the message and instantly redirect to login page, cy cant catch the message
    // cy.get('.message').should('contain', enMessages.navigation.log_in);

    // URL should change to login
    cy.url({ timeout: 8000 }).should('include', '/login');
  });
});
