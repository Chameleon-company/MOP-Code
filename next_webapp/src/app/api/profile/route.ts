import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";
import { validateProfileInput } from "@/app/api/library/validators";

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

function unauthorized() {
  return NextResponse.json(
    { success: false, message: "Unauthorised" },
    { status: 401 }
  );
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

// ---------------------------------------------------------------------------
// GET /api/profile
// ---------------------------------------------------------------------------

export async function GET(request: NextRequest) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();

  // Fetch profile from user_details
  const { data, error } = await supabase
    .from("user_details")
    .select(
      "id, user_id, first_name, last_name, age, gender, profile_img, created_at, updated_at"
    )
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    console.error("[GET /api/profile]", error);
    return serverError();
  }

  // Fetch email from user table
  const { data: userData, error: userError } = await supabase
    .from("user")
    .select("email")
    .eq("id", userId)
    .single();

  if (userError) {
    console.error("[GET /api/profile] user fetch error:", userError);
    return serverError();
  }

  // No profile row yet — return empty shell so the UI form still renders
  if (!data) {
    return NextResponse.json({
      success: true,
      data: {
        user_id: userId,
        first_name: null,
        last_name: null,
        age: null,
        gender: null,
        profile_img: null,
        email: userData?.email || null,
      },
    });
  }

  return NextResponse.json({
    success: true,
    data: {
      ...data,
      email: userData?.email || null,
    },
  });
}

// ---------------------------------------------------------------------------
// PUT /api/profile
// ---------------------------------------------------------------------------

interface ProfileUpdateBody {
  first_name?: string;
  last_name?: string;
  age?: number;
  gender?: string;
  profile_img?: string;
  email?: string;
}

export async function PUT(request: NextRequest) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();

  // --- Parse body -----------------------------------------------------------
  let body: ProfileUpdateBody;
  try {
    body = (await request.json()) as ProfileUpdateBody;
  } catch {
    return badRequest("Invalid JSON body");
  }

  // --- Validate using the validator utility ---------------------------------
  const validation = validateProfileInput(body);
  if (!validation.valid) {
    return badRequest("Validation failed", validation.errors);
  }

  const { first_name, last_name, age, gender, profile_img, email } = body;

  // --- Build update payload for user_details (only supplied fields) ---------
  const updates: Record<string, unknown> = {
    updated_at: new Date().toISOString(),
  };
  if (first_name !== undefined) updates.first_name = first_name.trim();
  if (last_name !== undefined) updates.last_name = last_name.trim();
  if (age !== undefined) updates.age = age;
  if (gender !== undefined) updates.gender = gender;
  if (profile_img !== undefined) updates.profile_img = profile_img;

  // Check if there are any user_details fields to update
  const hasUserDetailsUpdates = Object.keys(updates).length > 1;

  if (!hasUserDetailsUpdates && !email) {
    return badRequest("No updatable fields provided");
  }

  // --- Update email in 'user' table if provided ----------------------------
  if (email !== undefined) {
    const { error: emailError } = await supabase
      .from("user")
      .update({ email: email.trim() })
      .eq("id", userId);

    if (emailError) {
      console.error("[PUT /api/profile] email update error:", emailError);
      console.error("[PUT /api/profile] email update error details:", {
        message: emailError.message,
        code: emailError.code,
        details: emailError.details,
      });
      return serverError(`Failed to update email: ${emailError.message}`);
    }
  }

  let result: any = { email };

  // --- Update user_details if there are changes ---------------------------
  if (hasUserDetailsUpdates) {
    // --- Check if a profile row already exists for this user ------------------
    const { data: existing } = await supabase
      .from("user_details")
      .select("id")
      .eq("user_id", userId)
      .maybeSingle();

    if (existing) {
      // Row exists — UPDATE it
      const { data, error } = await supabase
        .from("user_details")
        .update(updates)
        .eq("user_id", userId)
        .select(
          "id, user_id, first_name, last_name, age, gender, profile_img, updated_at"
        )
        .single();

      if (error) {
        console.error("[PUT /api/profile] update error:", error);
        console.error("[PUT /api/profile] update error details:", {
          message: error.message,
          code: error.code,
          details: error.details,
        });
        return serverError(`Update failed: ${error.message}`);
      }
      result = { ...data, email };
    } else {
      // No row yet — INSERT a new one
      const { data, error } = await supabase
        .from("user_details")
        .insert({ user_id: userId, ...updates })
        .select(
          "id, user_id, first_name, last_name, age, gender, profile_img, updated_at"
        )
        .single();

      if (error) {
        console.error("[PUT /api/profile] insert error:", error);
        console.error("[PUT /api/profile] insert error details:", {
          message: error.message,
          code: error.code,
          details: error.details,
        });
        return serverError(`Insert failed: ${error.message}`);
      }
      result = { ...data, email };
    }
  }

  return NextResponse.json({
    success: true,
    message: "Profile updated successfully",
    data: result,
  });
}
