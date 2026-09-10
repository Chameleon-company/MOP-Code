import { Storage } from "@google-cloud/storage";

const storage = new Storage();

const PROXY_PREFIX = "/api/images/";

// Returns null for anything that isn't a proxy path — rows migrated from
// Supabase still hold absolute URLs and have no object in this bucket.
export function toGCSObjectPath(
  storedUrl: string | null | undefined,
): string | null {
  if (!storedUrl || !storedUrl.startsWith(PROXY_PREFIX)) return null;
  const objectPath = storedUrl.slice(PROXY_PREFIX.length);
  return objectPath.length > 0 ? objectPath : null;
}

// Best-effort cleanup. True if a delete was issued, false if there was no
// object path to act on. Callers must not let a failure fail the request.
export async function deleteImageFromGCS(
  storedUrl: string | null | undefined,
  bucketName: string,
): Promise<boolean> {
  const objectPath = toGCSObjectPath(storedUrl);
  if (!objectPath) return false;

  await storage
    .bucket(bucketName)
    .file(objectPath)
    .delete({ ignoreNotFound: true });

  return true;
}
