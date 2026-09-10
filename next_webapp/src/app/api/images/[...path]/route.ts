import { NextRequest, NextResponse } from "next/server";
import { Storage } from "@google-cloud/storage";
import { getImagesBucket } from "../../library/gcsBucket";

// Auth: Workload Identity Federation. No key file — the client picks up
// Application Default Credentials from the runtime environment. The Cloud
// Run service account needs roles/storage.objectViewer on this bucket.
const storage = new Storage();

// GET /api/images/[...path]
// Public — no auth required. Streams an object out of the private
// mop-images bucket (blogs/covers/..., gallery/...) so the browser never
// needs direct GCS access. See uploadImageToGCS.ts for the upload side.
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const objectPath = path.join("/");

  try {
    const [buffer] = await storage.bucket(getImagesBucket()).file(objectPath).download();

    return new NextResponse(buffer, {
      headers: {
        "Content-Type": "image/webp",
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error: any) {
    if (error?.code === 404) {
      return NextResponse.json({ success: false, message: "Image not found" }, { status: 404 });
    }
    console.error("[GET /api/images] error:", error);
    return NextResponse.json({ success: false, message: "Failed to fetch image" }, { status: 500 });
  }
}
