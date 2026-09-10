import { NextRequest, NextResponse } from "next/server";
import { getAuthUser } from "../library/auth";
import { uploadImageToGCS } from "../library/uploadImageToGCS";
import { getImagesBucket } from "../library/gcsBucket";

// ==============================
// POST /api/upload
// Upload an image to GCS (auth required)
// Body: multipart/form-data  →  file: File,  folder?: string,  bucket?: string
// Returns: { success, url }
// ==============================

// Still accepted so existing callers keep working, but everything now goes to
// the one images bucket under a folder prefix.
const ALLOWED_BUCKETS = ["category-images", "usecase-images"];

// Folder ends up in the object path, so keep it to a single plain segment.
function sanitizeFolder(folder: string): string | null {
  return /^[a-zA-Z0-9_-]+$/.test(folder) ? folder : null;
}

export async function POST(request: NextRequest) {
  const { userId, isAuthenticated } = getAuthUser(request);

  if (!isAuthenticated || !userId) {
    return NextResponse.json(
      { success: false, message: "Unauthorised" },
      { status: 401 }
    );
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const folderRaw = (formData.get("folder") as string | null) || "uploads";
    const bucketParam = (formData.get("bucket") as string | null) || "category-images";

    if (!ALLOWED_BUCKETS.includes(bucketParam)) {
      return NextResponse.json(
        { success: false, message: "Invalid storage bucket" },
        { status: 400 }
      );
    }

    const folder = sanitizeFolder(folderRaw);
    if (!folder) {
      return NextResponse.json(
        { success: false, message: "Invalid folder name" },
        { status: 400 }
      );
    }

    if (!file || file.size === 0) {
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

    // always .webp — uploadImageToGCS re-encodes
    const buffer = Buffer.from(await file.arrayBuffer());
    const filename = `${folder}/${userId}-${Date.now()}.webp`;

    let url: string;
    try {
      url = await uploadImageToGCS(buffer, filename, getImagesBucket());
    } catch (uploadError) {
      console.error("[POST /api/upload] upload error:", uploadError);
      return NextResponse.json(
        { success: false, message: "Upload failed" },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true, url });
  } catch (err) {
    console.error("[POST /api/upload] unexpected error:", err);
    return NextResponse.json(
      { success: false, message: "Internal server error" },
      { status: 500 }
    );
  }
}
