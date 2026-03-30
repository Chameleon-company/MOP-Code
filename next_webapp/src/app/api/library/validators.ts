/**
 * Reusable validation utilities for API inputs
 */

type ValidationError = { field: string; message: string };
type ValidationResult = {
  valid: boolean;
  errors: ValidationError[];
};

// ============================================================================
// STRING VALIDATORS
// ============================================================================

export function validateNonEmptyString(
  value: unknown,
  fieldName: string,
  maxLength = 255
): ValidationError | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return { field: fieldName, message: `${fieldName} must be a non-empty string` };
  }
  if (value.trim().length > maxLength) {
    return { field: fieldName, message: `${fieldName} must be ${maxLength} characters or fewer` };
  }
  return null;
}

export function validateEmail(email: unknown): ValidationError | null {
  if (typeof email !== "string") {
    return { field: "email", message: "Email must be a string" };
  }
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email.trim())) {
    return { field: "email", message: "Email format is invalid" };
  }
  return null;
}

// ============================================================================
// NUMBER VALIDATORS
// ============================================================================

export function validateInteger(
  value: unknown,
  fieldName: string,
  min?: number,
  max?: number
): ValidationError | null {
  if (!Number.isInteger(value)) {
    return { field: fieldName, message: `${fieldName} must be a whole number` };
  }

  if (min !== undefined && (value as number) < min) {
    return { field: fieldName, message: `${fieldName} must be at least ${min}` };
  }

  if (max !== undefined && (value as number) > max) {
    return { field: fieldName, message: `${fieldName} must be at most ${max}` };
  }

  return null;
}

// ============================================================================
// PASSWORD VALIDATORS
// ============================================================================

export function validatePasswordStrength(
  password: unknown,
  fieldName = "password"
): ValidationError | null {
  if (typeof password !== "string") {
    return { field: fieldName, message: `${fieldName} must be a string` };
  }

  if (password.length < 8) {
    return { field: fieldName, message: `${fieldName} must be at least 8 characters` };
  }

  if (!/[A-Z]/.test(password)) {
    return { field: fieldName, message: `${fieldName} must contain at least one uppercase letter` };
  }

  if (!/[a-z]/.test(password)) {
    return { field: fieldName, message: `${fieldName} must contain at least one lowercase letter` };
  }

  if (!/[0-9]/.test(password)) {
    return { field: fieldName, message: `${fieldName} must contain at least one digit` };
  }

  return null;
}

// ============================================================================
// ENUM VALIDATORS
// ============================================================================

export function validateEnumValue(
  value: unknown,
  allowedValues: Set<string> | string[],
  fieldName: string
): ValidationError | null {
  const allowed = Array.isArray(allowedValues) ? new Set(allowedValues) : allowedValues;
  if (!allowed.has(value as string)) {
    return {
      field: fieldName,
      message: `${fieldName} must be one of: ${Array.from(allowed).join(", ")}`,
    };
  }
  return null;
}

// ============================================================================
// PROFILE VALIDATORS
// ============================================================================

export function validateProfileInput(body: unknown): ValidationResult {
  const errors: ValidationError[] = [];

  if (typeof body !== "object" || body === null) {
    return {
      valid: false,
      errors: [{ field: "body", message: "Request body must be a valid object" }],
    };
  }

  const data = body as Record<string, unknown>;

  // Validate first_name
  if (data.first_name !== undefined) {
    const error = validateNonEmptyString(data.first_name, "first_name", 100);
    if (error) errors.push(error);
  }

  // Validate last_name
  if (data.last_name !== undefined) {
    const error = validateNonEmptyString(data.last_name, "last_name", 100);
    if (error) errors.push(error);
  }

  // Validate age
  if (data.age !== undefined) {
    const error = validateInteger(data.age, "age", 0, 150);
    if (error) errors.push(error);
  }

  // Validate gender
  if (data.gender !== undefined) {
    const genderError = validateEnumValue(
      data.gender,
      ["Male", "Female", "Other"],
      "gender"
    );
    if (genderError) errors.push(genderError);
  }

  // Validate profile_img
  if (data.profile_img !== undefined) {
    if (typeof data.profile_img !== "string") {
      errors.push({
        field: "profile_img",
        message: "profile_img must be a string",
      });
    }
  }

  // Validate email
  if (data.email !== undefined) {
    const error = validateEmail(data.email);
    if (error) errors.push(error);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// ============================================================================
// PASSWORD CHANGE VALIDATORS
// ============================================================================

export function validatePasswordChangeInput(body: unknown): ValidationResult {
  const errors: ValidationError[] = [];

  if (typeof body !== "object" || body === null) {
    return {
      valid: false,
      errors: [{ field: "body", message: "Request body must be a valid object" }],
    };
  }

  const data = body as Record<string, unknown>;

  // Current password
  if (!data.current_password || typeof data.current_password !== "string") {
    errors.push({
      field: "current_password",
      message: "current_password is required",
    });
  }

  // New password
  if (!data.new_password || typeof data.new_password !== "string") {
    errors.push({
      field: "new_password",
      message: "new_password is required",
    });
  } else {
    const strengthError = validatePasswordStrength(data.new_password, "new_password");
    if (strengthError) errors.push(strengthError);
  }

  // Confirm password
  if (!data.confirm_password || typeof data.confirm_password !== "string") {
    errors.push({
      field: "confirm_password",
      message: "confirm_password is required",
    });
  }

  // Semantic checks
  if (data.new_password && data.confirm_password) {
    if (data.new_password !== data.confirm_password) {
      errors.push({
        field: "confirm_password",
        message: "new_password and confirm_password do not match",
      });
    }

    if (data.new_password === data.current_password) {
      errors.push({
        field: "new_password",
        message: "new_password must differ from your current password",
      });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
