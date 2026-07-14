import sharp from "sharp";
import { supabase } from "@/library/supabaseClient";

const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

export async function uploadImageToStorage({
  file,
  bucket,
  folder,
  prefix,
  userId,
}: {
  file: File;
  bucket: string;
  folder: string;
  prefix: string;
  userId: number;
}) {
  if (!file) {
    throw new Error("No image file provided");
  }

  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    throw new Error("Invalid image type. Please upload JPEG, PNG, GIF, or WebP");
  }

  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error("Image size must be less than 5MB");
  }

  const arrayBuffer = await file.arrayBuffer();
  const inputBuffer = Buffer.from(arrayBuffer);

  const optimizedBuffer = await sharp(inputBuffer)
    .resize({
      width: 1600,
      withoutEnlargement: true,
    })
    .webp({
      quality: 80,
    })
    .toBuffer();

  const filename = `${prefix}-${userId}-${Date.now()}.webp`;
  const filePath = `${folder}/${filename}`;

  const { error } = await supabase.storage
    .from(bucket)
    .upload(filePath, optimizedBuffer, {
      contentType: "image/webp",
      upsert: false,
    });

  if (error) {
    throw new Error(`Image upload failed: ${error.message}`);
  }

  const { data } = supabase.storage.from(bucket).getPublicUrl(filePath);

  if (!data?.publicUrl) {
    throw new Error("Failed to generate image public URL");
  }

  return {
    filePath,
    publicUrl: data.publicUrl,
  };
}