import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/library/supabaseClient";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const blogId = Number(id);

  if (!Number.isFinite(blogId)) {
    return NextResponse.json({ success: false, message: "Invalid blog id" }, { status: 400 });
  }

  try {
    const { data: blog, error } = await supabase
      .from("blogs")
      .select("id, title, description, cover_img, published_date, content, created_by")
      .eq("id", blogId)
      .single();

    if (error || !blog) {
      return NextResponse.json({ success: false, message: "Blog not found" }, { status: 404 });
    }

    // Look up author name from user_details
    let authorName = "Admin";
    if (blog.created_by) {
      const { data: details } = await supabase
        .from("user_details")
        .select("first_name, last_name")
        .eq("user_id", blog.created_by)
        .single();

      if (details) {
        const name = [details.first_name, details.last_name].filter(Boolean).join(" ").trim();
        if (name) authorName = name;
      }
    }

    return NextResponse.json({
      success: true,
      data: { ...blog, author: authorName },
    });
  } catch (error) {
    console.error("[GET /api/home/blogs/[id]] error:", error);
    return NextResponse.json({ success: false, message: "Internal server error" }, { status: 500 });
  }
}
