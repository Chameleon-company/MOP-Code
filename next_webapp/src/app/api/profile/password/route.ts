import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { supabase } from "@/library/supabaseClient";
import { validatePasswordChangeInput } from "@/app/api/library/validators";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getUserId(request: NextRequest): number | null {
  const raw = request.headers.get("x-user-id");
  if (!raw) return null;
  const id = Number(raw);
  return Number.isFinite(id) ? id : null;
}

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
  const userId = getUserId(request);
  if (!userId) return unauthorized();

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
  const { data: user, error: fetchError } = await supabase
    .from("user")
    .select("password")
    .eq("id", userId)
    .maybeSingle();

  if (fetchError) {
    console.error("[PUT /api/profile/password] fetch error:", fetchError);
    return serverError();
  }

  // Return 401 (not 404) to avoid leaking whether a user ID exists
  if (!user) return unauthorized();

  // --- Verify the current password against the stored hash ------------------
  const matches = await bcrypt.compare(current_password!, user.password);
  if (!matches) return unauthorized("Current password is incorrect");

  // --- Hash the new password and save it ------------------------------------
  // 12 salt rounds = ~300ms on modern hardware, strong against brute force
  const newHash = await bcrypt.hash(new_password!, 12);

  const { error: updateError } = await supabase
    .from("user")
    .update({ password: newHash })
    .eq("id", userId);

  if (updateError) {
    console.error("[PUT /api/profile/password] update error:", updateError);
    return serverError();
  }

  return NextResponse.json({
    success: true,
    message: "Password updated successfully",
  });
}
