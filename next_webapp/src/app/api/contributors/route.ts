import { NextRequest, NextResponse } from "next/server";
import dbConnect from "@/lib/dbConnect";
import Contributor from "@/models/mongoose/Contributor";
import { getAuthUser } from "@/app/api/library/auth";
import { errorResponse } from "@/app/api/library/errorResponse";

// Map a Mongo document (or .lean() object) to the flat, snake_case shape the
// frontend expects plain string `id`, never a raw `_id`/`__v`.
function toContributorDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}

// ==============================
// GET /api/contributors
// List all contributors (PUBLIC the display page reads this unauthenticated)
// ==============================
export async function GET() {
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
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR");
  }
}

// ==============================
// POST /api/contributors
// Create a contributor (ADMIN ONLY)
// ==============================
export async function POST(request: NextRequest) {
  try {
    const { isAuthenticated, isAdmin } = getAuthUser(request);
    if (!isAuthenticated) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED");
    }
    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN");
    }

    const body = await request.json();

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
      return errorResponse(error.message, 400, "VALIDATION_ERROR");
    }
    console.error("Create Contributor Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR");
  }
}
