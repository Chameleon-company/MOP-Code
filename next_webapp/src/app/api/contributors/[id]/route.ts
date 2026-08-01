import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Contributor from "@/models/mongoose/Contributor";
import { getAuthUser } from "@/app/api/library/auth";
import { errorResponse } from "@/app/api/library/errorResponse";
import { toContributorDTO } from "@/app/api/library/contributorDto";

// ==============================
// GET /api/contributors/:id
// Fetch a single contributor (PUBLIC)
// ==============================
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid contributor ID", 400, "INVALID_ID", request);
    }

    await dbConnect();

    const contributor = await Contributor.findById(id).lean();
    if (!contributor) {
      return errorResponse("Contributor not found", 404, "NOT_FOUND", request);
    }

    return NextResponse.json({ success: true, data: toContributorDTO(contributor) });
  } catch (error) {
    console.error("Get Contributor Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request);
  }
}

// ==============================
// PUT /api/contributors/:id
// Update a contributor (ADMIN ONLY)
// ==============================
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { isAuthenticated, isAdmin } = getAuthUser(request);
    if (!isAuthenticated) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request);
    }
    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request);
    }

    const { id } = await params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid contributor ID", 400, "INVALID_ID", request);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return errorResponse("Invalid JSON in request body.", 400, "INVALID_JSON", request);
    }

    await dbConnect();

    const existing = await Contributor.findById(id);
    if (!existing) {
      return errorResponse("Contributor not found", 404, "NOT_FOUND", request);
    }

    existing.set({
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

    await existing.save();

    return NextResponse.json({ success: true, data: toContributorDTO(existing.toObject()) });
  } catch (error) {
    if (error instanceof Error && error.name === "ValidationError") {
      return errorResponse(error.message, 400, "VALIDATION_ERROR", request);
    }
    console.error("Update Contributor Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request);
  }
}

// ==============================
// DELETE /api/contributors/:id
// Delete a contributor (ADMIN ONLY)
// ==============================
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { isAuthenticated, isAdmin } = getAuthUser(request);
    if (!isAuthenticated) {
      return errorResponse("User not authenticated", 401, "UNAUTHORIZED", request);
    }
    if (!isAdmin) {
      return errorResponse("Forbidden - Admin only", 403, "FORBIDDEN", request);
    }

    const { id } = await params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid contributor ID", 400, "INVALID_ID", request);
    }

    await dbConnect();

    const deleted = await Contributor.findByIdAndDelete(id);
    if (!deleted) {
      return errorResponse("Contributor not found", 404, "NOT_FOUND", request);
    }

    return NextResponse.json({
      success: true,
      message: "Contributor deleted successfully",
    });
  } catch (error) {
    console.error("Delete Contributor Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request);
  }
}
