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
  return NextResponse.json({ success: false, message }, { status: 400 });
}

function notFound(message = "Blog not found") {
  return NextResponse.json({ success: false, message }, { status: 404 });
}

function serverError(message = "Internal server error") {
  return NextResponse.json({ success: false, message }, { status: 500 });
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const userId = getUserId(request);

  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const blogId = Number(params.id);

  if (!Number.isFinite(blogId)) {
    return badRequest("Invalid blog id");
  }

  try {
    const formData = await request.formData();

    const title = formData.get("title")?.toString().trim();
    const content = formData.get("content")?.toString().trim();
    const coverImg = formData.get("cover_img") as File | null;

    if (!title && !content && !coverImg) {
      return badRequest("At least one field is required to update");
    }

    const updatePayload: {
      title?: string;
      content?: string;
      cover_img?: string;
      updated_at: string;
    } = {
      updated_at: new Date().toISOString(),
    };

    if (title) updatePayload.title = title;
    if (content) updatePayload.content = content;

    if (coverImg) {
      const allowedTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];

      if (!allowedTypes.includes(coverImg.type)) {
        return badRequest("Invalid cover image type. Please upload JPEG, PNG, GIF, or WebP");
      }

      if (coverImg.size > 5 * 1024 * 1024) {
        return badRequest("Cover image size must be less than 5MB");
      }

      const buffer = await coverImg.arrayBuffer();
      const fileExtension = coverImg.name.split(".").pop();
      const filename = `blog-cover-${userId}-${Date.now()}.${fileExtension}`;
      const filePath = `blogs/covers/${filename}`;

      const { error: uploadError } = await supabase.storage
        .from("blog-images")
        .upload(filePath, buffer, {
          contentType: coverImg.type,
          upsert: false,
        });

      if (uploadError) {
        console.error("[PUT /api/blogs/[id]] upload error:", uploadError);
        return serverError(`Cover image upload failed: ${uploadError.message}`);
      }

      const { data: publicUrlData } = supabase.storage
        .from("blog-images")
        .getPublicUrl(filePath);

      if (!publicUrlData?.publicUrl) {
        return serverError("Failed to generate cover image URL");
      }

      updatePayload.cover_img = publicUrlData.publicUrl;
    }

    const { data: blog, error } = await supabase
      .from("blogs")
      .update(updatePayload)
      .eq("id", blogId)
      .select("*")
      .single();

    if (error) {
      console.error("[PUT /api/blogs/[id]] update error:", error);
      return serverError(`Failed to update blog: ${error.message}`);
    }

    if (!blog) return notFound();

    return NextResponse.json(
      {
        success: true,
        message: "Blog updated successfully",
        blog,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("[PUT /api/blogs/[id]] error:", error);
    return serverError("Failed to update blog");
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const userId = getUserId(request);

  if (!userId) return unauthorized();
  if (!isAdmin(request)) return forbidden();

  const blogId = Number(params.id);

  if (!Number.isFinite(blogId)) {
    return badRequest("Invalid blog id");
  }

  try {
    const { error } = await supabase
      .from("blogs")
      .delete()
      .eq("id", blogId);

    if (error) {
      console.error("[DELETE /api/blogs/[id]] delete error:", error);
      return serverError(`Failed to delete blog: ${error.message}`);
    }

    return NextResponse.json(
      {
        success: true,
        message: "Blog deleted successfully",
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("[DELETE /api/blogs/[id]] error:", error);
    return serverError("Failed to delete blog");
  }
}