import { Storage } from "@google-cloud/storage";
import sharp from "sharp";

// Auth: Workload Identity Federation. No key file — the client picks up
// Application Default Credentials from the runtime environment.
const storage = new Storage();

const MAX_WIDTH = 1600;
const WEBP_QUALITY = 80;

/**
 * Resize/re-encode an image to WebP and upload it to the given GCS bucket.
 * The bucket (mop-images, australia-southeast1) is private — objects aren't
 * publicly readable, so this returns the app-relative proxy path
 * (/api/images/<filename>, served by src/app/api/images/[...path]/route.ts)
 * rather than a storage.googleapis.com URL. The Cloud Run service account
 * reads the object server-side via its own project-level access; the
 * browser never talks to GCS directly.
 */
export async function uploadImageToGCS(
  buffer: Buffer,
  filename: string,
  bucketName: string,
): Promise<string> {
  const optimizedBuffer = await sharp(buffer)
    .resize({
      width: MAX_WIDTH,
      withoutEnlargement: true,
    })
    .webp({
      quality: WEBP_QUALITY,
    })
    .toBuffer();

  const file = storage.bucket(bucketName).file(filename);

  await file.save(optimizedBuffer, {
    contentType: "image/webp",
    resumable: false,
  });

  return `/api/images/${filename}`;
}
