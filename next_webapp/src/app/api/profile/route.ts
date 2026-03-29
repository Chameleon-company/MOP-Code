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

  const { data, error } = await supabase
    .from("user_details")
    .select(
      "id, user_id, first_name, last_name, age, gender, profile_img, created_at, updated_at"
    )
    .eq("user_id", userId)
    .maybeSingle(); // returns null (not an error) when no row is found

  if (error) {
    console.error("[GET /api/profile]", error);
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
      },
    });
  }

  return NextResponse.json({ success: true, data });
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

  const { first_name, last_name, age, gender, profile_img } = body;

  // --- Build update payload (only supplied fields) --------------------------
  const updates: Record<string, unknown> = {
    updated_at: new Date().toISOString(),
  };
  if (first_name !== undefined) updates.first_name = first_name.trim();
  if (last_name !== undefined) updates.last_name = last_name.trim();
  if (age !== undefined) updates.age = age;
  if (gender !== undefined) updates.gender = gender;
  if (profile_img !== undefined) updates.profile_img = profile_img;

  if (Object.keys(updates).length === 1) {
    // Only updated_at was added — nothing real to update
    return badRequest("No updatable fields provided");
  }

  // --- Check if a profile row already exists for this user ------------------
  const { data: existing } = await supabase
    .from("user_details")
    .select("id")
    .eq("user_id", userId)
    .maybeSingle();

  let result;

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
      return serverError();
    }
    result = data;
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
      return serverError();
    }
    result = data;
  }

  return NextResponse.json({
    success: true,
    message: "Profile updated successfully",
    data: result,
  });
}
