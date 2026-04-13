import { NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import {
    UpdateCategoryDTO,
    validateUpdateCategory,
    sanitizeCategoryInput,
} from "@/models/Category";
import { errorResponse } from "@/app/api/library/errorResponse";
import { getAuthUser } from "@/app/api/library/auth";

// ==============================
// PUT /api/categories/:id
// Update Category (ADMIN ONLY)
// ==============================

export async function PUT(
    request: Request,
    { params }: { params: { id: string } }
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

        const categoryId = Number(params.id);

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