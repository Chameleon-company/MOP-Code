import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Category from "@/models/mongoose/Category";
import UseCase from "@/models/mongoose/UseCase";
import {
    UpdateCategoryDTO,
    validateUpdateCategory,
    sanitizeCategoryInput,
} from "@/types/category";
import { errorResponse } from "@/app/api/library/errorResponse";
import { getAuthUser } from "@/app/api/library/auth";

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
    const { _id, __v, ...rest } = doc;
    return { id: _id.toString(), ...rest };
}

function escapeRegex(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

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

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid category ID", 400, "INVALID_ID");
    }

    await dbConnect();

    const category = await Category.findById(id).lean();

    if (!category) {
      return errorResponse("Category not found", 404, "NOT_FOUND");
    }

    return NextResponse.json({ success: true, data: toDTO(category) });
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
    const { userId, isAdmin } = getAuthUser(request as any);

    try {
        // ==============================
        // 1. Auth check
        // ==============================

        if (!userId) {
            return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request, userId);
        }

        if (!isAdmin) {
            return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request, userId);
        }

        const { id } = await params;

        if (!mongoose.Types.ObjectId.isValid(id)) {
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
            return errorResponse(validationError, 400, "VALIDATION_ERROR", request, userId);
        }

        await dbConnect();

        // ==============================
        // 4. Check if category exists
        // ==============================
        const existing = await Category.findById(id);

        if (!existing) {
            return errorResponse("Category not found", 404, "NOT_FOUND");
        }

        // ==============================
        // 5. Check duplicate (only if category_name is updating)
        // ==============================

        if (cleanData.category_name) {
            const duplicate = await Category.findOne({
                _id: { $ne: id },
                category_name: {
                    $regex: `^${escapeRegex(cleanData.category_name)}$`,
                    $options: "i",
                },
            }).lean();

            if (duplicate) {
                return errorResponse(
                    "Category with this name already exists",
                    400,
                    "DUPLICATE_CATEGORY",
                    request,
                    userId
                );
            }
        }

        // ==============================
        // 6. Update category
        // ==============================
        existing.set(cleanData);
        await existing.save();

        // ==============================
        // 7. Success response
        // ==============================
        return NextResponse.json({
            success: true,
            message: "Category updated successfully",
            data: toDTO(existing.toObject()),
        });
    } catch (error) {
        console.error("Update Category Error:", error);

        return errorResponse(
            "Internal Server Error",
            500,
            "INTERNAL_ERROR",
            request,
            userId
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
  const { userId, isAdmin } = getAuthUser(request);

  try {
    // 1. Auth check

    if (!userId) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request, userId);
    }

    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request, userId);
    }

    const { id } = await params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid category ID", 400, "INVALID_ID");
    }

    await dbConnect();

    // 2. Check category exists
    const existingCategory = await Category.findById(id).lean();

    if (!existingCategory) {
      return errorResponse("Category not found", 404, "CATEGORY_NOT_FOUND");
    }

    // 3. Count how many use cases are using this category
    const count = await UseCase.countDocuments({ "category.id": id });

    if (count > 0) {
      return NextResponse.json(
        {
          success: false,
          message: `${count} use case(s) are currently assigned to this category. Please change them before deleting the category.`,
          code: "CATEGORY_IN_USE",
          data: {
            assigned_usecase_count: count,
            category_id: (existingCategory as any)._id.toString(),
            category_name: (existingCategory as any).category_name,
          },
        },
        { status: 400 }
      );
    }

    // 4. Delete category
    await Category.findByIdAndDelete(id);

    // 5. Success response
    return NextResponse.json(
      {
        success: true,
        message: "Category deleted successfully",
        data: {
          id: (existingCategory as any)._id.toString(),
          category_name: (existingCategory as any).category_name,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Delete Category Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request, userId);
  }
}