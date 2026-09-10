// Resolved per call rather than as a module constant: Cloud Run injects this
// at deploy time, so it isn't present while `next build` runs.
export function getImagesBucket(): string {
  const raw = process.env.GCS_IMAGES_BUCKET;

  if (!raw || raw.trim() === "") {
    throw new Error("GCS_IMAGES_BUCKET is not set");
  }

  // cloudbuild.yaml sets this as gs://<bucket>; the client wants a bare name.
  return raw.trim().replace(/^gs:\/\//i, "").replace(/\/+$/, "");
}
