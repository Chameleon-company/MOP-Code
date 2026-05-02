import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";

function getUserId(request: NextRequest): number | null {
  const raw = request.headers.get("x-user-id");
  if (!raw) return null;

  const id = Number(raw);
  return Number.isFinite(id) ? id : null;
}

function isAdmin(request: NextRequest): boolean {
  const role = request.headers.get("x-user-role");
  const roleId = request.headers.get("x-user-role-id");

  return role?.toLowerCase() === "admin" || roleId === "1";
}

function unauthorized() {
  return NextResponse.json(
    { success: false, message: "Unauthorised" },
    { status: 401 }
  );
}

function forbidden() {
  return NextResponse.json(
    { success: false, message: "Forbidden - Admin only" },
    { status: 403 }
  );
}

function badRequest(message: string) {
  return NextResponse.json(
    { success: false, message },
    { status: 400 }
  );
}

function serverError(message = "Internal server error") {
  return NextResponse.json(
    { success: false, message },
    { status: 500 }
  );
}

// add blog
export async function POST(request: NextRequest) {
  const userId = getUserId(request);

  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  try {
    const formData = await request.formData();

    const title = formData.get("title")?.toString().trim();
    const content = formData.get("content")?.toString().trim();
    const coverImage = formData.get("cover_img") as File | null;

    if (!title || !content) {
      return badRequest("Title and content are required");
    }

    if (!coverImage) {
      return badRequest("Cover image is required");
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];

    if (!allowedTypes.includes(coverImage.type)) {
      return badRequest("Invalid cover image type. Please upload JPEG, PNG, GIF, or WebP");
    }

    if (coverImage.size > 5 * 1024 * 1024) {
      return badRequest("Cover image size must be less than 5MB");
    }

    const buffer = await coverImage.arrayBuffer();

    const fileExtension = coverImage.name.split(".").pop();
    const filename = `blog-cover-${userId}-${Date.now()}.${fileExtension}`;
    const filePath = `blogs/covers/${filename}`;

    const { error: uploadError } = await supabase.storage
      .from("blog-images")
      .upload(filePath, buffer, {
        contentType: coverImage.type,
        upsert: false,
      });

    if (uploadError) {
      console.error("[POST /api/blogs] upload error:", uploadError);
      return serverError(`Cover image upload failed: ${uploadError.message}`);
    }

    const { data: publicUrlData } = supabase.storage
      .from("blog-images")
      .getPublicUrl(filePath);

    if (!publicUrlData?.publicUrl) {
      return serverError("Failed to generate cover image URL");
    }

    const { data: blog, error: insertError } = await supabase
      .from("blogs")
      .insert({
        title,
        content,
        cover_img: publicUrlData.publicUrl,
        created_by: userId,
        updated_at: new Date().toISOString(),
      })
      .select("*")
      .single();

    if (insertError) {
      console.error("[POST /api/blogs] insert error:", insertError);
      return serverError(`Failed to create blog: ${insertError.message}`);
    }

    return NextResponse.json(
      {
        success: true,
        message: "Blog created successfully",
        blog,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("[POST /api/blogs] error:", error);
    return serverError("Failed to create blog");
  }
}

//list blogs
export async function GET() {
  try {
    const { data: blogs, error } = await supabase
      .from("blogs")
      .select(`
        id,
        created_at,
        title,
        content,
        cover_img,
        updated_at,
        created_by
      `)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("[GET /api/blogs] error:", error);
      return serverError(`Failed to fetch blogs: ${error.message}`);
    }

    return NextResponse.json(
      {
        success: true,
        blogs,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("[GET /api/blogs] error:", error);
    return serverError("Failed to fetch blogs");
  }
}