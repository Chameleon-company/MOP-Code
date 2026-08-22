// models/Category.ts

// ==============================
// 1. Base Category Model (DB Shape)
// ==============================
export interface Category {
  _id: string;
  legacy_id: string | null;
  category_name: string;
  description: string | null;
  cover_img: string | null;
  created_at: string;
  updated_at: string;
  created_by: number | null;
}

// ==============================
// 2. DTOs (Data Transfer Objects)
// ==============================

// For creating a category
export interface CreateCategoryDTO {
  category_name: string;
  description?: string;
  cover_img?: string;
}

// For updating a category
export interface UpdateCategoryDTO {
  category_name?: string;
  description?: string;
  cover_img?: string;
}

// ==============================
// 3. Validation Functions
// ==============================

// Validate Create Category
export function validateCreateCategory(data: CreateCategoryDTO): string | null {
  if (!data.category_name || data.category_name.trim().length === 0) {
    return "Category name is required";
  }

  if (data.category_name.length > 100) {
    return "Category name must be less than 100 characters";
  }

  if (data.description && data.description.length > 500) {
    return "Description must be less than 500 characters";
  }

  return null;
}

// Validate Update Category
export function validateUpdateCategory(data: UpdateCategoryDTO): string | null {
  if (
    data.category_name !== undefined &&
    data.category_name.trim().length === 0
  ) {
    return "Category name cannot be empty";
  }

  if (data.category_name && data.category_name.length > 100) {
    return "Category name must be less than 100 characters";
  }

  if (data.description && data.description.length > 500) {
    return "Description must be less than 500 characters";
  }

  return null;
}

// ==============================
// 4. Helper: sanitize input
// ==============================

export function sanitizeCategoryInput<T extends object>(data: T): T {
  const sanitized: any = {};

  for (const key in data) {
    const value = (data as any)[key];

    if (typeof value === "string") {
      sanitized[key] = value.trim();
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}