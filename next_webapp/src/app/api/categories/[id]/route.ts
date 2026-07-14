import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import {
    UpdateCategoryDTO,
    validateUpdateCategory,
    sanitizeCategoryInput,
} from "@/models/Category";
import { errorResponse } from "@/app/api/library/errorResponse";
import { getAuthUser } from "@/app/api/library/auth";

// ==============================
// GET /api/categories/:id
// Fetch single category (auth required)
// ==============================

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { userId } = getAuthUser(request);
    if (!userId) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
    }

    const { id } = await params;
    const categoryId = Number(id);

    if (!categoryId || Number.isNaN(categoryId)) {
      return errorResponse("Invalid category ID", 400, "INVALID_ID");
    }

    const { data, error } = await supabase
      .from("categories")
      .select("*")
      .eq("id", categoryId)
      .single();

    if (error || !data) {
      return errorResponse("Category not found", 404, "NOT_FOUND");
    }

    return NextResponse.json({ success: true, data });
  } catch (error) {
    console.error("Get Category Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR");
  }
}

// ==============================
// PUT /api/categories/:id
// Update Category (ADMIN ONLY)
// ==============================

export async function PUT(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        // ==============================
        // 1. Auth check
        // ==============================
        const { userId, isAdmin } = getAuthUser(request as any);

        if (!userId) {
            return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
        }

        if (!isAdmin) {
            return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN");
        }

        const { id } = await params;
        const categoryId = Number(id);

        if (!categoryId) {
            return errorResponse("Invalid category ID", 400, "INVALID_ID");
        }

        // ==============================
        // 2. Parse + sanitize input
        // ==============================
        const body: UpdateCategoryDTO = await request.json();
        const cleanData = sanitizeCategoryInput(body);

        // ==============================
        // 3. Validate input
        // ==============================
        const validationError = validateUpdateCategory(cleanData);

        if (validationError) {
            return errorResponse(validationError, 400, "VALIDATION_ERROR");
        }

        // ==============================
        // 4. Check if category exists
        // ==============================
        const { data: existing, error: fetchError } = await supabase
            .from("categories")
            .select("*")
            .eq("id", categoryId)
            .single();

        if (fetchError || !existing) {
            return errorResponse("Category not found", 404, "NOT_FOUND");
        }

        // ==============================
        // 5. Check duplicate (only if category_name is updating)
        // ==============================

        if (cleanData.category_name) {
            const { data: duplicate, error: duplicateError } = await supabase
                .from("categories")
                .select("id")
                .ilike("category_name", cleanData.category_name)
                .neq("id", categoryId) // exclude current record
                .maybeSingle();

            if (duplicateError) {
                console.error("Duplicate check error:", duplicateError);
                return errorResponse(
                    "Failed to validate category",
                    500,
                    "DB_CHECK_ERROR"
                );
            }

            if (duplicate) {
                return errorResponse(
                    "Category with this name already exists",
                    400,
                    "DUPLICATE_CATEGORY"
                );
            }
        }

        // ==============================
        // 6. Update category
        // ==============================
        const { data, error } = await supabase
            .from("categories")
            .update({
                ...cleanData,
                updated_at: new Date().toISOString(),
            })
            .eq("id", categoryId)
            .select()
            .single();

        if (error) {
            console.error("Update error:", error);
            return errorResponse("Failed to update category", 500, "DB_UPDATE_ERROR");
        }

        // ==============================
        // 7. Fetch updated user info
        // ==============================
        const { data: updatedUser } = await supabase
            .from("user")
            .select("id, email, role_id")
            .eq("id", data.created_by)
            .single();

        // ==============================
        // 8. Success response
        // ==============================
        return NextResponse.json({
            success: true,
            message: "Category updated successfully",
            data: {
                ...data,
                created_by_user: updatedUser || null,
            },
        });
    } catch (error) {
        console.error("Update Category Error:", error);

        return errorResponse(
            "Internal Server Error",
            500,
            "INTERNAL_ERROR"
        );
    }
}

// ==============================
// DELETE /api/categories/:id
// Delete Category (ADMIN ONLY)
// ==============================

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    // 1. Auth check
    const { userId, isAdmin } = getAuthUser(request);

    if (!userId) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
    }

    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN");
    }

    const { id } = await params;
    const categoryId = Number(id);

    if (!categoryId || Number.isNaN(categoryId)) {
      return errorResponse("Invalid category ID", 400, "INVALID_ID");
    }

    // 2. Check category exists
    const { data: existingCategory, error: categoryError } = await supabase
      .from("categories")
      .select("id, category_name")
      .eq("id", categoryId)
      .single();

    if (categoryError || !existingCategory) {
      return errorResponse("Category not found", 404, "CATEGORY_NOT_FOUND");
    }

    // 3. Count how many use cases are using this category
    const { count, error: countError } = await supabase
      .from("usecases")
      .select("*", { count: "exact", head: true })
      .eq("category_id", categoryId);

    if (countError) {
      console.error("Use case count error:", countError);
      return errorResponse(
        "Failed to validate category usage",
        500,
        "USAGE_CHECK_ERROR"
      );
    }

    if ((count ?? 0) > 0) {
      return NextResponse.json(
        {
          success: false,
          message: `${count} use case(s) are currently assigned to this category. Please change them before deleting the category.`,
          code: "CATEGORY_IN_USE",
          data: {
            assigned_usecase_count: count,
            category_id: categoryId,
            category_name: existingCategory.category_name,
          },
        },
        { status: 400 }
      );
    }

    // 4. Delete category
    const { error: deleteError } = await supabase
      .from("categories")
      .delete()
      .eq("id", categoryId);

    if (deleteError) {
      console.error("Delete category error:", deleteError);
      return errorResponse(
        "Failed to delete category",
        500,
        "DELETE_ERROR"
      );
    }

    // 5. Success response
    return NextResponse.json(
      {
        success: true,
        message: "Category deleted successfully",
        data: {
          id: existingCategory.id,
          category_name: existingCategory.category_name,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Delete Category Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR");
  }
}
