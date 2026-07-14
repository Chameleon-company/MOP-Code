import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";

function getUserId(request: NextRequest): number | null {
  const raw = request.headers.get("x-user-id");
  if (!raw) return null;
  const id = Number(raw);
  return Number.isFinite(id) ? id : null;
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

function badRequest(message: string) {
  return NextResponse.json({ success: false, message }, { status: 400 });
}

export async function POST(request: NextRequest) {
  const userId = getUserId(request);
  if (!userId) return unauthorized();

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
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

    // Convert file to buffer
    const buffer = await file.arrayBuffer();

    // Generate unique filename
    const filename = `profile-${userId}-${Date.now()}.${file.name.split(".").pop()}`;
    const filePath = `profiles/${userId}/${filename}`;

    // Upload to Supabase Storage
    const { data, error } = await supabase.storage
      .from("profile-images")
      .upload(filePath, buffer, {
        contentType: file.type,
        upsert: false,
      });

    if (error) {
      console.error("[POST /api/profile/upload-image] upload error:", error);
      return serverError(`Upload failed: ${error.message}`);
    }

    // Get public URL
    const { data: publicUrlData } = supabase.storage
      .from("profile-images")
      .getPublicUrl(filePath);

    if (!publicUrlData?.publicUrl) {
      return serverError("Failed to generate public URL");
    }

    console.log("[POST /api/profile/upload-image] success:", publicUrlData.publicUrl);

    return NextResponse.json({
      success: true,
      message: "Image uploaded successfully",
      imageUrl: publicUrlData.publicUrl,
    });
  } catch (error) {
    console.error("[POST /api/profile/upload-image] error:", error);
    return serverError("Failed to process upload");
  }
}
