describe('Chatbot Functionality Tests', () => {
    // Visit the page where the chatbot is located before each test
    beforeEach(() => {
        cy.visit('/'); // leave as is since chatbot is in the homepage
    });

    it('should open the chatbot when clicked', () => {
        // Check that the chatbot is closed initially
        cy.get('.chat-window').should('not.exist');
        
        // Click the chatbot toggle button
        cy.get('.toggle-btn').click();
        
        // Assert that the chatbot window is visible
        cy.get('.chat-window').should('be.visible');
    });

    it('should send a message and receive a bot response', () => {
        // Open the chatbot
        cy.get('.toggle-btn').click();
        
        // Type a message in the input field
        cy.get('input.textarea1').type('Hello');
        
        // Click the send button
        cy.get('.send-icon').click();
        
        // Check that the user's message is displayed in the chat window
        cy.get('.messages').should('contain', 'Hello');
        
        // Check that the bot responds after sending a message
        cy.get('.messages').should('contain', "Sorry, I didn't understand that. Can you try rephrasing?");
    });

    it('should navigate to the correct page when a recognized command is entered', () => {
        // Open the chatbot
        cy.get('.toggle-btn').click();
        
        // Type a recognized command, e.g., 'login page'
        cy.get('input.textarea1').type('login page');
        
        // Click the send button
        cy.get('.send-icon').click();
        
        // Verify that the bot gives a response about redirecting
        cy.get('.messages').should('contain', 'Understood. Redirecting to the right page.');

        // Simulate the redirect (you can mock the `router.push` or test if URL changes)
        cy.url().should('include', '/en/login'); // change if needed
    });

    it('should display an error message for unrecognized input', () => {
        // Open the chatbot
        cy.get('.toggle-btn').click();
        
        // Type an unrecognized message
        cy.get('input.textarea1').type('random message');
        
        // Click the send button
        cy.get('.send-icon').click();
        
        // Check that the bot displays the fallback message
        cy.get('.messages').should('contain', "Sorry, I didn't understand that. Can you try rephrasing?");
    });
});
