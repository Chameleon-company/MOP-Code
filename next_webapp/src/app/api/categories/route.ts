import { NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import {
    CreateCategoryDTO,
    validateCreateCategory,
    sanitizeCategoryInput,
} from "@/models/Category";
import { errorResponse } from "@/app/api/library/errorResponse";
import { getAuthUser } from "@/app/api/library/auth";
import { NextRequest } from "next/server";

// ==============================
// POST /api/categories
// Create Category (ADMIN ONLY)
// ==============================

export async function POST(request: NextRequest) {
    try {
        // ==============================
        // 1. Check Admin Authorization
        // ==============================
        const { userId, isAuthenticated, isAdmin } = getAuthUser(request);

        if (!isAuthenticated) {
            return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
        }

        if (!isAdmin) {
            return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN");
        }


        // ==============================
        // 2. Parse & Sanitize Input
        // ==============================
        const body: CreateCategoryDTO = await request.json();
        const cleanData = sanitizeCategoryInput(body);

        // ==============================
        // 3. Validate Input
        // ==============================
        const validationError = validateCreateCategory(cleanData);

        if (validationError) {
            return errorResponse(validationError, 400, "VALIDATION_ERROR");
        }

        const { category_name, description } = cleanData;

        // ==============================
        // 4. Check duplicate category
        // ==============================

        const { data: existingCategory, error: checkError } = await supabase
            .from("categories")
            .select("id")
            .ilike("category_name", category_name)
            .maybeSingle();

        if (checkError) {
            console.error("Duplicate Check Error:", checkError);
            return errorResponse(
                "Failed to validate category",
                500,
                "DB_CHECK_ERROR"
            );
        }

        if (existingCategory) {
            return errorResponse(
                "Category already exists",
                400,
                "DUPLICATE_CATEGORY"
            );
        }

        // ==============================
        // 5. Insert into Supabase
        // ==============================
        const { data, error } = await supabase
            .from("categories")
            .insert([
                {
                    category_name,
                    description: description ?? null,
                    created_by: Number(userId),
                },
            ])
            .select()
            .single();


        const { data: createdUser, error: userError } = await supabase
            .from("user")
            .select("id, email, role_id")
            .eq("id", data.created_by)
            .single();

        if (userError) {
            console.error("User fetch error:", userError);
        }

        if (error) {
            console.error("Supabase Insert Error:", error);
            return errorResponse(
                "Failed to create category",
                500,
                "DB_INSERT_ERROR"
            );
        }

        // ==============================
        // 5. Success Response
        // ==============================
        return NextResponse.json(
            {
                success: true,
                message: "Category created successfully",
                data: {
                    ...data,
                    created_by_user: createdUser || null,
                },
            },
            { status: 201 }
        );
    } catch (error) {
        console.error("Create Category Error:", error);

        return errorResponse(
            "Internal Server Error",
            500,
            "INTERNAL_ERROR"
        );
    }
}