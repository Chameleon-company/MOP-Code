import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import dbConnect from "@/lib/dbConnect";
import User from "@/models/mongoose/User";
import { getAuthUser } from "@/app/api/library/auth";
import { validatePasswordChangeInput } from "@/app/api/library/validators";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function badRequest(message: string, errors?: { field: string; message: string }[]) {
  const response: any = { success: false, message };
  if (errors?.length) {
    response.errors = errors;
  }
  return NextResponse.json(response, { status: 400 });
}

function unauthorized(message = "Unauthorised") {
  return NextResponse.json({ success: false, message }, { status: 401 });
}

function serverError() {
  return NextResponse.json(
    { success: false, message: "Internal server error" },
    { status: 500 }
  );
}

// ---------------------------------------------------------------------------
// PUT /api/profile/password
// ---------------------------------------------------------------------------

export async function PUT(request: NextRequest) {
  const { userId, isAuthenticated } = getAuthUser(request);
  if (!isAuthenticated || !userId) return unauthorized();

  // --- Parse body -----------------------------------------------------------
  let body: {
    current_password?: string;
    new_password?: string;
    confirm_password?: string;
  };
  try {
    body = await request.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // --- Validate using the validator utility ---------------------------------
  const validation = validatePasswordChangeInput(body);
  if (!validation.valid) {
    return badRequest("Validation failed", validation.errors);
  }

  const { current_password, new_password, confirm_password } = body;

  // --- Fetch the stored bcrypt hash from the user table ---------------------
  try {
    await dbConnect();

    const user = await User.findById(userId).select("password");

    // Return 401 (not 404) to avoid leaking whether a user ID exists
    if (!user) return unauthorized();

    // --- Verify the current password against the stored hash ------------------
    const matches = await bcrypt.compare(current_password!, user.password);
    if (!matches) return unauthorized("Current password is incorrect");

    // --- Hash the new password and save it ------------------------------------
    // 12 salt rounds = ~300ms on modern hardware, strong against brute force
    const newHash = await bcrypt.hash(new_password!, 12);

    user.password = newHash;
    await user.save();

    return NextResponse.json({
      success: true,
      message: "Password updated successfully",
    });
  } catch (error) {
    console.error("[PUT /api/profile/password] error:", error);
    return serverError();
  }
}