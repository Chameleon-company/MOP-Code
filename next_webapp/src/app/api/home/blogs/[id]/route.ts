import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Blog from "@/models/mongoose/Blog";
import User from "@/models/mongoose/User";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!mongoose.Types.ObjectId.isValid(id)) {
    return NextResponse.json({ success: false, message: "Invalid blog id" }, { status: 400 });
  }

  try {
    await dbConnect();

    const blog = await Blog.findById(id)
      .select("title description cover_img published_date content created_by")
      .lean();

    if (!blog) {
      return NextResponse.json({ success: false, message: "Blog not found" }, { status: 404 });
    }

    // Look up author name from the User's embedded profile
    let authorName = "Admin";
    if (blog.created_by) {
      const author = await User.findById(blog.created_by)
        .select("profile.first_name profile.last_name")
        .lean();

      if (author) {
        const name = [author.profile?.first_name, author.profile?.last_name]
          .filter(Boolean)
          .join(" ")
          .trim();
        if (name) authorName = name;
      }
    }

    const { _id, __v, ...rest } = blog as any;

    return NextResponse.json({
      success: true,
      data: { id: _id.toString(), ...rest, author: authorName },
    });
  } catch (error) {
    console.error("[GET /api/home/blogs/[id]] error:", error);
    return NextResponse.json({ success: false, message: "Internal server error" }, { status: 500 });
  }
}
