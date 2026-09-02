// Authentication flow E2E tests — covers login, signup, forgot-password, change-password, and OTP-verification
// form validation, error states, success states, and navigation.
// Admin access-control and sidebar navigation tests are in admin.cy.ts
//
// Selectors:
//   - Email input:      cy.get('input[type="email"]')
//   - Password input:   cy.get('input[name="password"]')
//   - Submit button:    cy.get('button[type="submit"]')
//   - Show password:    cy.get('button[aria-label="Show password"]')

// 1. Login Page
describe('Login Page', () => {
  beforeEach(() => {
    cy.visit('/login');
  });

  it('renders the login form with all required fields', () => {
    // Email, password, and submit button must all be visible
    cy.get('input[type="email"]').should('be.visible');
    cy.get('input[name="password"]').should('be.visible');
    cy.get('button[type="submit"]').should('be.visible');
  });

  it('shows the app logo on the login page', () => {
    // The Melbourne Open Data logo should be visible
    cy.get('img[alt="Melbourne Open Data logo"]').should('be.visible');
  });

  it('submitting empty form does not call the API', () => {
    cy.intercept('POST', '/api/auth/login').as('loginAPI');
    // Click submit without filling any fields
    cy.get('button[type="submit"]').click();
    // Native HTML5 required validation prevents submission — API should not be called
    cy.get('@loginAPI.all').should('have.length', 0);
  });

  it('shows an error message for invalid credentials', () => {
    cy.intercept('POST', '/api/auth/login', {
      statusCode: 401,
      body: { message: 'Invalid email or password' },
    }).as('loginFail');

    // Fill in wrong credentials
    cy.get('input[type="email"]').type('wrong@example.com');
    cy.get('input[name="password"]').type('wrongpassword');
    // Submit the form
    cy.get('button[type="submit"]').click();

    // Wait for the failed API response
    cy.wait('@loginFail');
    // Error message should be shown to the user
    cy.contains('Invalid email or password').should('be.visible');
  });

  it('password visibility toggle shows/hides password', () => {
    // Type a password into the field
    cy.get('input[name="password"]').type('secret123');
    // Password should be hidden by default
    cy.get('input[name="password"]').should('have.attr', 'type', 'password');

    // Click show password button
    cy.get('button[aria-label="Show password"]').click();
    // Password should now be visible as text
    cy.get('input[name="password"]').should('have.attr', 'type', 'text');

    // Click hide password button
    cy.get('button[aria-label="Hide password"]').click();
    // Password should be hidden again
    cy.get('input[name="password"]').should('have.attr', 'type', 'password');
  });

  it('has a link to the sign up page', () => {
    // Use href selector to scope to the actual <a> element — avoids matching button text
    cy.get('a[href*="/signup"]').should('be.visible');
    cy.get('a[href*="/signup"]').click();
    cy.url({ timeout: 8000 }).should('include', '/signup');
  });

  it('has a forgot password link', () => {
    // Use href selector to scope to the actual <a> element — avoids matching surrounding text
    cy.get('a[href*="/forgot-password"]').should('be.visible');
    cy.get('a[href*="/forgot-password"]').click();
    cy.url({ timeout: 8000 }).should('include', '/forgot-password');
  });

  it('shows client-side error when only email is filled', () => {
    // The login form uses noValidate and checks both fields before calling the API
    cy.get('input[type="email"]').type('user@example.com');
    // Leave password empty and submit
    cy.get('button[type="submit"]').click();
    // Client-side validation message should appear
    cy.contains('Please fill in both fields').should('be.visible');
  });

  it('shows a generic error when the network request fails', () => {
    cy.intercept('POST', '/api/auth/login', { forceNetworkError: true }).as('loginNetworkError');

    // Fill in credentials
    cy.get('input[type="email"]').type('user@example.com');
    cy.get('input[name="password"]').type('Password1!');
    // Submit the form
    cy.get('button[type="submit"]').click();

    // Generic error should be shown
    cy.contains('Something went wrong. Please try again.').should('be.visible');
  });

  it('shows a loading spinner while the login request is in-flight', () => {
    // Delay the API response so we can observe the loading state
    cy.intercept('POST', '/api/auth/login', {
      statusCode: 200,
      body: { success: true, data: { userId: '1', token: 'tok', roleId: 2 } },
      delay: 1000,
    }).as('loginSlow');

    cy.get('input[type="email"]').type('user@example.com');
    cy.get('input[name="password"]').type('Password1!');
    cy.get('button[type="submit"]').click();

    // While the request is in-flight the button should show "Signing in..." text
    cy.contains('Signing in...').should('be.visible');
    // Wait for the delayed response to fully resolve before the test ends
    cy.wait('@loginSlow');
  });
});

// Helper: fills firstName, lastName, and email — shared by most signup validation tests
const fillSignupBase = (email = 'jane@example.com') => {
  cy.get('input[name="firstName"]').type('Jane');
  cy.get('input[name="lastName"]').type('Doe');
  cy.get('input[name="email"]').type(email);
};

// 2. Signup Page
describe('Signup Page', () => {
  beforeEach(() => {
    cy.visit('/signup');
  });

  it('renders the signup form with all required fields', () => {
    // All required input fields should be visible
    cy.get('input[name="firstName"]').should('be.visible');
    cy.get('input[name="lastName"]').should('be.visible');
    cy.get('input[name="email"]').should('be.visible');
    cy.get('input[name="password"]').should('be.visible');
  });

  it('shows validation errors when submitting an empty form', () => {
    // Click submit without filling any fields
    cy.get('button[type="submit"]').click();
    // First name validation error should appear
    cy.contains('Please enter your first name').should('be.visible');
  });

  it('shows password strength indicator when typing a password', () => {
    // Type a weak password
    cy.get('input[name="password"]').type('weak');
    cy.contains('Weak').should('be.visible');

    // Type a strong password
    cy.get('input[name="password"]').clear().type('StrongPass1!');
    cy.contains('Strong').should('be.visible');
  });

  it('has a link back to the login page', () => {
    // Use href selector to scope to the actual <a> element
    cy.get('a[href*="/login"]').click();
    // Should navigate back to /login
    cy.url({ timeout: 8000 }).should('include', '/login');
  });

  it('shows last name validation error when only first name is filled', () => {
    // Fill first name but intentionally skip last name
    cy.get('input[name="firstName"]').type('Jane');
    cy.get('input[name="email"]').type('jane@example.com');
    cy.get('input[name="password"]').type('StrongPass1!');
    cy.get('input[name="confirmPassword"]').type('StrongPass1!');
    cy.get('button[type="submit"]').click();
    // Last name validation error should appear
    cy.contains('Please enter your last name').should('be.visible');
  });

  it('shows email validation error for an invalid email format', () => {
    // Override email with an invalid value — all other fields are valid
    fillSignupBase('not-an-email');
    cy.get('input[name="password"]').type('StrongPass1!');
    cy.get('input[name="confirmPassword"]').type('StrongPass1!');
    cy.get('button[type="submit"]').click();
    // Email validation error should appear
    cy.contains('Please enter a valid email address').should('be.visible');
  });

  it('shows error when password is too short', () => {
    fillSignupBase();
    // Enter a password shorter than 8 characters
    cy.get('input[name="password"]').type('Ab1!');
    cy.get('input[name="confirmPassword"]').type('Ab1!');
    cy.get('button[type="submit"]').click();
    cy.contains('Your password must be at least 8 characters long').should('be.visible');
  });

  it('shows error when password lacks complexity requirements', () => {
    fillSignupBase();
    // 8+ chars but missing special character
    cy.get('input[name="password"]').type('Abcdefgh1');
    cy.get('input[name="confirmPassword"]').type('Abcdefgh1');
    cy.get('button[type="submit"]').click();
    cy.contains('Password must include uppercase, lowercase, number, and special character').should('be.visible');
  });

  it('shows error when passwords do not match', () => {
    fillSignupBase();
    cy.get('input[name="password"]').type('StrongPass1!');
    // Confirm password differs
    cy.get('input[name="confirmPassword"]').type('DifferentPass1!');
    cy.get('button[type="submit"]').click();
    cy.contains('Passwords do not match').should('be.visible');
  });

  it('shows error when the email is already registered', () => {
    cy.intercept('POST', '/api/auth/signup', {
      statusCode: 400,
      body: { message: 'User already exists', code: 'USER_EXISTS' },
    }).as('signupDuplicate');

    // Override email to simulate a pre-existing account
    fillSignupBase('existing@example.com');
    cy.get('input[name="password"]').type('StrongPass1!');
    cy.get('input[name="confirmPassword"]').type('StrongPass1!');
    cy.get('button[type="submit"]').click();

    cy.wait('@signupDuplicate');
    cy.contains('User already exists').should('be.visible');
  });

  it('shows a generic error when the signup network request fails', () => {
    cy.intercept('POST', '/api/auth/signup', { forceNetworkError: true }).as('signupNetworkError');

    fillSignupBase();
    cy.get('input[name="password"]').type('StrongPass1!');
    cy.get('input[name="confirmPassword"]').type('StrongPass1!');
    cy.get('button[type="submit"]').click();

    cy.contains('Something went wrong. Please try again later.').should('be.visible');
  });
});


// 3. Forgot Password Page
describe('Forgot Password Page', () => {
  beforeEach(() => {
    cy.visit('/forgot-password');
  });

  it('renders the forgot-password form with email input and submit button', () => {
    // Heading, email input, and submit button should all be visible
    cy.contains('Forgot Password').should('be.visible');
    cy.get('input[type="email"]').should('be.visible');
    cy.get('button[type="submit"]').should('be.visible');
  });

  it('shows an error when submitting with an empty email', () => {
    // Submit without entering an email
    cy.get('button[type="submit"]').click();
    // Client-side validation message should appear
    cy.contains('Please enter your email address').should('be.visible');
  });

  it('shows an error when the API returns INVALID_EMAIL', () => {
    cy.intercept('POST', '/api/auth/forgot-password', {
      statusCode: 400,
      body: { code: 'INVALID_EMAIL', message: 'A valid email address is required' },
    }).as('forgotInvalid');

    cy.get('input[type="email"]').type('bad-email');
    cy.get('button[type="submit"]').click();

    cy.wait('@forgotInvalid');
    // Mapped error message from ERROR_MESSAGES
    cy.contains('Please enter a valid email address').should('be.visible');
  });

  it('shows a success message when the email is accepted', () => {
    cy.intercept('POST', '/api/auth/forgot-password', {
      statusCode: 200,
      body: { success: true, message: 'If this email exists, a temporary password has been sent' },
    }).as('forgotSuccess');

    cy.get('input[type="email"]').type('user@example.com');
    cy.get('button[type="submit"]').click();

    cy.wait('@forgotSuccess');
    cy.contains('If this email exists, a temporary password has been sent').should('be.visible');
  });

  it('shows a generic error when the network request fails', () => {
    cy.intercept('POST', '/api/auth/forgot-password', { forceNetworkError: true }).as('forgotNetworkError');

    cy.get('input[type="email"]').type('user@example.com');
    cy.get('button[type="submit"]').click();

    cy.contains('Something went wrong. Please try again.').should('be.visible');
  });

  it('has a "Back to Sign In" link that navigates to /login', () => {
    // Use contains('a', ...) to scope to the actual <a> element
    cy.contains('a', 'Back to Sign In').should('be.visible');
    cy.contains('a', 'Back to Sign In').click();
    cy.url({ timeout: 8000 }).should('include', '/login');
  });
});

// 4. Change Password Page
describe('Change Password Page', () => {
  const validUrl = '/change-password?email=user@example.com';

  beforeEach(() => {
    // Most tests use validUrl — individual tests can override with cy.visit() if needed
    cy.visit(validUrl);
  });

  it('renders the form when a valid email query param is present', () => {
    // Heading and all three password inputs should be visible
    cy.contains('Change Password').should('be.visible');
    cy.get('#tempPassword').should('be.visible');
    cy.get('#newPassword').should('be.visible');
    cy.get('#confirmNewPassword').should('be.visible');
    cy.get('button[type="submit"]').should('be.visible');
  });

  it('shows "Invalid or expired reset link" when no email query param is provided', () => {
    // Override beforeEach navigation — visit without the email query param
    cy.visit('/change-password');
    // The error banner should appear and the submit button should be disabled
    cy.contains('Invalid or expired reset link').should('be.visible');
    cy.get('button[type="submit"]').should('be.disabled');
  });

  it('shows an error when submitting with empty fields', () => {
    cy.get('button[type="submit"]').click();
    cy.contains('All fields are required').should('be.visible');
  });

  it('shows an error when the new password is shorter than 8 characters', () => {
    cy.get('#tempPassword').type('TempPass1');
    cy.get('#newPassword').type('Ab1!');
    cy.get('#confirmNewPassword').type('Ab1!');
    cy.get('button[type="submit"]').click();
    cy.contains('New password must be at least 8 characters').should('be.visible');
  });

  it('shows an error when new passwords do not match', () => {
    cy.get('#tempPassword').type('TempPass1');
    cy.get('#newPassword').type('NewPassword1!');
    cy.get('#confirmNewPassword').type('DifferentPassword1!');
    cy.get('button[type="submit"]').click();
    cy.contains('New passwords do not match').should('be.visible');
  });

  it('shows an error when the temporary password is incorrect', () => {
    cy.intercept('POST', '/api/auth/reset-password', {
      statusCode: 401,
      body: { code: 'INVALID_TEMP_PASSWORD', message: 'Invalid temporary password' },
    }).as('resetInvalidTemp');

    cy.get('#tempPassword').type('WrongTemp1');
    cy.get('#newPassword').type('NewPassword1!');
    cy.get('#confirmNewPassword').type('NewPassword1!');
    cy.get('button[type="submit"]').click();

    cy.wait('@resetInvalidTemp');
    cy.contains('Temporary password is incorrect').should('be.visible');
  });

  it('shows an error when the new password is the same as the temp password', () => {
    cy.intercept('POST', '/api/auth/reset-password', {
      statusCode: 400,
      body: { code: 'SAME_AS_TEMP_PASSWORD', message: 'New password must be different from temporary password' },
    }).as('resetSameAsTemp');

    cy.get('#tempPassword').type('TempPass1!');
    cy.get('#newPassword').type('TempPass1!');
    cy.get('#confirmNewPassword').type('TempPass1!');
    cy.get('button[type="submit"]').click();

    cy.wait('@resetSameAsTemp');
    cy.contains('New password must be different from your temporary password').should('be.visible');
  });

  it('shows an error when no account is found for the email', () => {
    cy.intercept('POST', '/api/auth/reset-password', {
      statusCode: 401,
      body: { code: 'INVALID_CREDENTIALS', message: 'Invalid credentials' },
    }).as('resetInvalidCreds');

    cy.get('#tempPassword').type('TempPass1');
    cy.get('#newPassword').type('NewPassword1!');
    cy.get('#confirmNewPassword').type('NewPassword1!');
    cy.get('button[type="submit"]').click();

    cy.wait('@resetInvalidCreds');
    cy.contains('No account found for that email address').should('be.visible');
  });

  it('shows success message on successful password change', () => {
    cy.intercept('POST', '/api/auth/reset-password', {
      statusCode: 200,
      body: { success: true, message: 'Password reset successfully' },
    }).as('resetSuccess');

    cy.get('#tempPassword').type('TempPass1');
    cy.get('#newPassword').type('NewPassword1!');
    cy.get('#confirmNewPassword').type('NewPassword1!');
    cy.get('button[type="submit"]').click();

    cy.wait('@resetSuccess');
    cy.contains('Password changed successfully').should('be.visible');
  });

  it('password visibility toggles work for all three fields', () => {
    // All three fields should start as type="password"
    cy.get('#tempPassword').should('have.attr', 'type', 'password');
    cy.get('#newPassword').should('have.attr', 'type', 'password');
    cy.get('#confirmNewPassword').should('have.attr', 'type', 'password');

    // Re-query each button individually — avoids stale DOM refs after React re-renders on each click
    cy.get('button[aria-label="Show password"]').eq(0).click();
    cy.get('button[aria-label="Show password"]').eq(0).click();
    cy.get('button[aria-label="Show password"]').eq(0).click();

    // All three fields should now be type="text"
    cy.get('#tempPassword').should('have.attr', 'type', 'text');
    cy.get('#newPassword').should('have.attr', 'type', 'text');
    cy.get('#confirmNewPassword').should('have.attr', 'type', 'text');
  });

  it('has a "Back to Sign In" link that navigates to /login', () => {
    // Use contains('a', ...) to scope to the actual <a> element
    cy.contains('a', 'Back to Sign In').should('be.visible');
    cy.contains('a', 'Back to Sign In').click();
    cy.url({ timeout: 8000 }).should('include', '/login');
  });
});

// 5. OTP Verification Page
describe('OTP Verification Page', () => {
  beforeEach(() => {
    cy.visit('/otp_verification');
  });

  it('renders the OTP verification form with all fields', () => {
    // Heading and key inputs should be visible
    cy.contains('OTP VERIFICATION').should('be.visible');
    cy.get('input[placeholder="Four Digit Code"]').should('be.visible');
    cy.get('input[placeholder="New Password"]').should('be.visible');
    cy.get('input[placeholder="Confirm Password"]').should('be.visible');
    cy.contains('Reset Password').should('be.visible');
  });

  it('shows "Passwords do not match" when new and confirm passwords differ', () => {
    // Type mismatching passwords
    cy.get('input[placeholder="New Password"]').type('Password1!');
    cy.get('input[placeholder="Confirm Password"]').type('Different1!');
    // Inline error should appear
    cy.contains('Passwords do not match').should('be.visible');
  });

  it('has a "Sign up" link', () => {
    cy.contains('Sign up').should('be.visible');
  });
});
