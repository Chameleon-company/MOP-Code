import { NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Category from "@/models/mongoose/Category";
import {
    CreateCategoryDTO,
    validateCreateCategory,
    sanitizeCategoryInput,
} from "@/types/category";
import { errorResponse } from "@/app/api/library/errorResponse";
import { getAuthUser } from "@/app/api/library/auth";
import { NextRequest } from "next/server";
import logger from "@/utils/logger";

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
    const { _id, __v, ...rest } = doc;
    return { id: _id.toString(), ...rest };
}

// Escape regex metacharacters so user input can't be used to build an
// unintended (or catastrophic) regular expression.
function escapeRegex(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ==============================
// POST /api/categories
// Create Category (ADMIN ONLY)
// ==============================

export async function POST(request: NextRequest) {
    const { userId, isAuthenticated, isAdmin } = getAuthUser(request);

    try {
        // ==============================
        // 1. Check Admin Authorization
        // ==============================

        if (!isAuthenticated) {
            return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request, userId);
        }

        if (!isAdmin) {
            return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request, userId);
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
            return errorResponse(validationError, 400, "VALIDATION_ERROR", request, userId);
        }

        const { category_name, description, cover_img } = cleanData;

        await dbConnect();

        // ==============================
        // 4. Check duplicate category
        // ==============================

        const existingCategory = await Category.findOne({
            category_name: { $regex: `^${escapeRegex(category_name)}$`, $options: "i" },
        }).lean();

        if (existingCategory) {
            return errorResponse(
                "Category already exists",
                400,
                "DUPLICATE_CATEGORY",
                request,
                userId
            );
        }

        // ==============================
        // 5. Insert into MongoDB
        // ==============================
        const createdBy =
            userId && mongoose.Types.ObjectId.isValid(String(userId))
                ? String(userId)
                : null;

        const created = await Category.create({
            category_name,
            description: description ?? null,
            cover_img: cover_img ?? null,
            created_by: createdBy,
        });

        // ==============================
        // 6. Success Response
        // ==============================
        return NextResponse.json(
            {
                success: true,
                message: "Category created successfully",
                data: toDTO(created.toObject()),
            },
            { status: 201 }
        );
    } catch (error) {
        logger.error(`Create Category Error: ${error instanceof Error ? error.message : String(error)}`);

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
// GET /api/categories
// Fetch all categories (with optional search filter)
// ==============================
export async function GET(request: NextRequest) {
  try {
    // 1. Auth check
    const { userId } = getAuthUser(request);
    if (!userId) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
    }

    // 2. Get query params
    const { searchParams } = new URL(request.url);
    const search = searchParams.get("search");
    const page = Math.max(1, parseInt(searchParams.get("page") ?? "1", 10) || 1);
    const pageSize = Math.max(1, parseInt(searchParams.get("pageSize") ?? "10", 10) || 10);
    const skip = (page - 1) * pageSize;

    await dbConnect();

    // 3. Build filter
    const filter: Record<string, unknown> = {};
    if (search && search.trim().length > 0) {
      filter.category_name = { $regex: escapeRegex(search.trim()), $options: "i" };
    }

    // 4. Execute query
    const [data, total] = await Promise.all([
      Category.find(filter)
        .sort({ created_at: -1 })
        .skip(skip)
        .limit(pageSize)
        .lean(),
      Category.countDocuments(filter),
    ]);

    // 5. Return response
    return NextResponse.json({
      success: true,
      data: data.map(toDTO),
      count: data.length,
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });

  } catch (error) {
    logger.error(`[GET /api/categories] unexpected error: ${error instanceof Error ? error.message : String(error)}`);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR");
  }
}