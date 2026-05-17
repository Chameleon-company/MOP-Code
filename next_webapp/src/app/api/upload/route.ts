import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";

// ==============================
// POST /api/upload
// Upload an image to Supabase Storage (auth required)
// Body: multipart/form-data  →  file: File,  folder?: string
// Returns: { success, url }
// ==============================

export async function POST(request: NextRequest) {
  const userIdRaw = request.headers.get("x-user-id");
  const userId = userIdRaw ? Number(userIdRaw) : null;

  if (!userId) {
    return NextResponse.json(
      { success: false, message: "Unauthorised" },
      { status: 401 }
    );
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const folder = (formData.get("folder") as string | null) || "uploads";
    const bucketParam = (formData.get("bucket") as string | null) || "category-images";

    const ALLOWED_BUCKETS = ["category-images", "usecase-images"];
    if (!ALLOWED_BUCKETS.includes(bucketParam)) {
      return NextResponse.json(
        { success: false, message: "Invalid storage bucket" },
        { status: 400 }
      );
    }

    if (!file) {
      return NextResponse.json(
        { success: false, message: "No file provided" },
        { status: 400 }
      );
    }

    const allowed = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (!allowed.includes(file.type)) {
      return NextResponse.json(
        { success: false, message: "Only JPEG, PNG, GIF or WebP images are allowed" },
        { status: 400 }
      );
    }

    if (file.size > 5 * 1024 * 1024) {
      return NextResponse.json(
        { success: false, message: "File must be under 5 MB" },
        { status: 400 }
      );
    }

    const ext = file.name.split(".").pop();
    const filename = `${folder}/${userId}-${Date.now()}.${ext}`;
    const buffer = await file.arrayBuffer();

    const { error: uploadError } = await supabase.storage
      .from(bucketParam)
      .upload(filename, buffer, { contentType: file.type, upsert: false });

    if (uploadError) {
      console.error("[POST /api/upload] upload error:", uploadError);
      return NextResponse.json(
        { success: false, message: "Upload failed: " + uploadError.message },
        { status: 500 }
      );
    }

    const { data: publicData } = supabase.storage
      .from(bucketParam)
      .getPublicUrl(filename);

    return NextResponse.json({ success: true, url: publicData.publicUrl });
  } catch (err) {
    console.error("[POST /api/upload] unexpected error:", err);
    return NextResponse.json(
      { success: false, message: "Internal server error" },
      { status: 500 }
    );
  }
}
