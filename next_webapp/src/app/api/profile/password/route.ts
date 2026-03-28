import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { supabase } from "@/library/supabaseClient";
 
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
 
function getUserId(request: NextRequest): number | null {
  const raw = request.headers.get("x-user-id");
  if (!raw) return null;
  const id = Number(raw);
  return Number.isFinite(id) ? id : null;
}
 
function badRequest(message: string) {
  return NextResponse.json({ success: false, message }, { status: 400 });
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
// Password strength rules
// Adjust to match your team's security policy.
// ---------------------------------------------------------------------------
 
function validatePasswordStrength(password: string): string | null {
  if (password.length < 8)
    return "Password must be at least 8 characters";
  if (!/[A-Z]/.test(password))
    return "Password must contain at least one uppercase letter";
  if (!/[a-z]/.test(password))
    return "Password must contain at least one lowercase letter";
  if (!/[0-9]/.test(password))
    return "Password must contain at least one digit";
  return null;
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
 
  const { current_password, new_password, confirm_password } = body;
 
  // --- Field presence checks ------------------------------------------------
  if (!current_password || typeof current_password !== "string")
    return badRequest("current_password is required");
  if (!new_password || typeof new_password !== "string")
    return badRequest("new_password is required");
  if (!confirm_password || typeof confirm_password !== "string")
    return badRequest("confirm_password is required");
 
  // --- Semantic checks ------------------------------------------------------
  if (new_password !== confirm_password)
    return badRequest("new_password and confirm_password do not match");
 
  if (new_password === current_password)
    return badRequest("new_password must differ from your current password");
 
  const strengthError = validatePasswordStrength(new_password);
  if (strengthError) return badRequest(strengthError);
 
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
  const matches = await bcrypt.compare(current_password, user.password);
  if (!matches) return unauthorized("Current password is incorrect");
 
  // --- Hash the new password and save it ------------------------------------
  // 12 salt rounds = ~300ms on modern hardware, strong against brute force
  const newHash = await bcrypt.hash(new_password, 12);
 
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