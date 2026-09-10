import { NextRequest, NextResponse } from "next/server";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";
import { getImagesBucket } from "@/app/api/library/gcsBucket";

// TODO: prefix not confirmed with the GCP owner yet — follows the old
// Supabase layout for now.
const PROFILE_IMAGE_PREFIX = "profiles";

function unauthorized() {
  return NextResponse.json(
    { success: false, message: "Unauthorised" },
    { status: 401 }
  );
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

function badRequest(message: string) {
  return NextResponse.json({ success: false, message }, { status: 400 });
}

// ==============================
// POST /api/profile/upload-image
// Upload a profile image to GCS (auth required)
// Body: multipart/form-data → file: File
// Returns: { success, message, imageUrl }
//
// Only stores the image. PUT /api/profile saves the URL onto the user.
// ==============================

export async function POST(request: NextRequest) {
  const { userId, isAuthenticated } = getAuthUser(request);
  if (!isAuthenticated || !userId) return unauthorized();

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;

    if (!file || file.size === 0) {
      return badRequest("No file provided");
    }

    // Validate file type
    const allowedTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      return badRequest("Invalid file type. Please upload JPEG, PNG, GIF, or WebP");
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      return badRequest("File size must be less than 5MB");
    }

    const buffer = Buffer.from(await file.arrayBuffer());

    // always .webp — uploadImageToGCS re-encodes
    const filename = `${PROFILE_IMAGE_PREFIX}/${userId}/profile-${userId}-${Date.now()}.webp`;

    let imageUrl: string;
    try {
      imageUrl = await uploadImageToGCS(buffer, filename, getImagesBucket());
    } catch (uploadError) {
      console.error("[POST /api/profile/upload-image] upload error:", uploadError);
      return serverError("Profile image upload failed");
    }

    return NextResponse.json({
      success: true,
      message: "Image uploaded successfully",
      imageUrl,
    });
  } catch (error) {
    console.error("[POST /api/profile/upload-image] error:", error);
    return serverError("Failed to process upload");
  }
}
