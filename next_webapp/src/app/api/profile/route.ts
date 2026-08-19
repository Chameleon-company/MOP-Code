import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import { validateProfileInput } from "@/app/api/library/validators";
import dbConnect from "@/lib/dbConnect";
import User from "@/models/mongoose/User";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getUserId(request: NextRequest): string | null {
  const raw = request.headers.get("x-user-id");

  if (!raw || !mongoose.Types.ObjectId.isValid(raw)) {
    return null;
  }

  return raw;
}

function badRequest(
  message: string,
  errors?: { field: string; message: string }[],
) {
  const response: any = { success: false, message };

  if (errors?.length) {
    response.errors = errors;
  }

  return NextResponse.json(response, { status: 400 });
}

function unauthorized() {
  return NextResponse.json(
    { success: false, message: "Unauthorised" },
    { status: 401 },
  );
}

function serverError(message = "Internal server error") {
  return NextResponse.json(
    { success: false, message },
    { status: 500 },
  );
}

// ---------------------------------------------------------------------------
// GET /api/profile
// ---------------------------------------------------------------------------

export async function GET(request: NextRequest) {
  const userId = getUserId(request);

  if (!userId) {
    return unauthorized();
  }

  try {
    await dbConnect();

    // Fetch the user and its embedded profile object.
    const user = await User.findById(userId)
      .select("email profile created_at updated_at")
      .exec();

    if (!user) {
      return unauthorized();
    }

    // No embedded profile yet — return an empty shell so the UI still renders.
    if (!user.profile) {
      return NextResponse.json({
        success: true,
        data: {
          user_id: String(user._id),
          first_name: null,
          last_name: null,
          age: null,
          gender: null,
          profile_img: null,
          email: user.email || null,
        },
      });
    }

    return NextResponse.json({
      success: true,
      data: {
        id: String(user._id),
        user_id: String(user._id),
        first_name: user.profile.first_name ?? null,
        last_name: user.profile.last_name ?? null,
        age: user.profile.age ?? null,
        gender: user.profile.gender ?? null,
        profile_img: user.profile.profile_img ?? null,
        created_at: (user as any).created_at ?? null,
        updated_at:
          user.profile.updated_at ??
          (user as any).updated_at ??
          null,
        email: user.email || null,
      },
    });
  } catch (error) {
    console.error("[GET /api/profile]", error);
    return serverError();
  }
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

interface ProfileUpdateResult {
  email: string | undefined;
  id?: string;
  user_id?: string;
  first_name?: string | null;
  last_name?: string | null;
  age?: number | null;
  gender?: string | null;
  profile_img?: string | null;
  updated_at?: Date | null;
}

export async function PUT(request: NextRequest) {
  const userId = getUserId(request);

  if (!userId) {
    return unauthorized();
  }

  // --- Parse body -----------------------------------------------------------
  let body: ProfileUpdateBody;

  try {
    body = (await request.json()) as ProfileUpdateBody;
  } catch {
    return badRequest("Invalid JSON body");
  }

  // --- Validate using the validator utility --------------------------------
  const validation = validateProfileInput(body);

  if (!validation.valid) {
    return badRequest("Validation failed", validation.errors);
  }

  const {
    first_name,
    last_name,
    age,
    gender,
    profile_img,
    email,
  } = body;

  // Check whether any embedded profile fields were supplied.
  const hasProfileUpdates =
    first_name !== undefined ||
    last_name !== undefined ||
    age !== undefined ||
    gender !== undefined ||
    profile_img !== undefined;

  if (!hasProfileUpdates && !email) {
    return badRequest("No updatable fields provided");
  }

  try {
    await dbConnect();

    const user = await User.findById(userId).exec();

    if (!user) {
      return unauthorized();
    }

    // --- Update email if provided -------------------------------------------
    if (email !== undefined) {
      user.set("email", email.trim());
    }

    // --- Update embedded profile if fields were provided --------------------
    if (hasProfileUpdates) {
      // Existing migrated users may not have a profile object yet.
      if (!user.profile) {
        user.set("profile", {});
      }

      if (first_name !== undefined) {
        user.set("profile.first_name", first_name.trim());
      }

      if (last_name !== undefined) {
        user.set("profile.last_name", last_name.trim());
      }

      if (age !== undefined) {
        user.set("profile.age", age);
      }

      if (gender !== undefined) {
        user.set("profile.gender", gender);
      }

      if (profile_img !== undefined) {
        user.set("profile.profile_img", profile_img);
      }

      user.set("profile.updated_at", new Date());
    }

    await user.save();

    let result: ProfileUpdateResult = { email };

    if (hasProfileUpdates) {
      result = {
        id: String(user._id),
        user_id: String(user._id),
        first_name: user.profile?.first_name ?? null,
        last_name: user.profile?.last_name ?? null,
        age: user.profile?.age ?? null,
        gender: user.profile?.gender ?? null,
        profile_img: user.profile?.profile_img ?? null,
        updated_at: user.profile?.updated_at ?? null,
        email,
      };
    }

    return NextResponse.json({
      success: true,
      message: "Profile updated successfully",
      data: result,
    });
  } catch (error) {
    console.error("[PUT /api/profile] update error:", error);

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    return serverError(`Update failed: ${message}`);
  }
}