import { NextRequest, NextResponse } from "next/server";
import dbConnect from "@/lib/dbConnect";
import Contributor from "@/models/mongoose/Contributor";
import { getAuthUser } from "@/app/api/library/auth";
import { errorResponse } from "@/app/api/library/errorResponse";
import { toContributorDTO } from "@/app/api/library/contributorDto";

// ==============================
// GET /api/contributors
// List all contributors (PUBLIC the display page reads this unauthenticated)
// ==============================
export async function GET(request: NextRequest) {
  try {
    await dbConnect();

    const contributors = await Contributor.find({})
      .sort({ year: -1, trimester: 1, display_order: 1 })
      .lean();

    return NextResponse.json({
      success: true,
      data: contributors.map(toContributorDTO),
    });
  } catch (error) {
    console.error("List Contributors Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request);
  }
}

// ==============================
// POST /api/contributors
// Create a contributor (ADMIN ONLY)
// ==============================
export async function POST(request: NextRequest) {
  const { userId, isAuthenticated, isAdmin } = getAuthUser(request);
  try {
    if (!isAuthenticated) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request, userId);
    }
    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request, userId);
    }

    let body;
    try {
      body = await request.json();
    } catch (error) {
      if (error instanceof SyntaxError) {
        return errorResponse("Invalid JSON in request body.", 400, "INVALID_JSON", request, userId);
      }
      throw error;
    }

    await dbConnect();

    const created = await Contributor.create({
      name: body.name,
      year: body.year,
      trimester: body.trimester,
      contributor_type: body.contributor_type,
      team: body.team ?? null,
      position: body.position ?? null,
      level: body.level ?? null,
      display_order: body.display_order ?? 0,
      is_active: body.is_active ?? true,
    });

    return NextResponse.json(
      { success: true, data: toContributorDTO(created.toObject()) },
      { status: 201 },
    );
  } catch (error) {
    if (error instanceof Error && error.name === "ValidationError") {
      return errorResponse(error.message, 400, "VALIDATION_ERROR", request, userId);
    }
    console.error("Create Contributor Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request, userId);
  }
}
