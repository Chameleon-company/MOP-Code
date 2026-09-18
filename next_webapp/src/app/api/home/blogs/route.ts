import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import Blog from "@/models/mongoose/Blog";

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
function toDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);

    await dbConnect();

    /** Random picks for “Continue exploring” (blog detail). Excludes current id. */
    if (url.searchParams.get("recommend") === "1") {
      const excludeId = url.searchParams.get("excludeId");
      const take = Math.min(
        10,
        Math.max(1, parseInt(url.searchParams.get("take") ?? "3", 10) || 3)
      );

      const filter: Record<string, unknown> = {};
      if (excludeId && mongoose.Types.ObjectId.isValid(excludeId)) {
        filter._id = { $ne: excludeId };
      }

      const data = await Blog.find(filter)
        .select("title description cover_img published_date")
        .limit(800)
        .lean();

      const pool = [...data];
      shuffleInPlace(pool);

      return NextResponse.json({
        success: true,
        data: pool.slice(0, take).map(toDTO),
      });
    }

    const page = Math.max(1, parseInt(url.searchParams.get("page") ?? "1", 10) || 1);
    const pageSize = Math.min(
      50,
      Math.max(1, parseInt(url.searchParams.get("pageSize") ?? "9", 10) || 9)
    );
    const skip = (page - 1) * pageSize;

    const search = url.searchParams.get("search")?.trim() ?? "";
    const searchBy = url.searchParams.get("search_by")?.trim() ?? "title";

    const filter: Record<string, unknown> = {};
    if (search) {
      if (searchBy === "content") {
        filter.$or = [
          { description: { $regex: escapeRegex(search), $options: "i" } },
          { content: { $regex: escapeRegex(search), $options: "i" } },
        ];
      } else {
        filter.title = { $regex: escapeRegex(search), $options: "i" };
      }
    }

    const [data, total] = await Promise.all([
      Blog.find(filter)
        .select("title description cover_img published_date created_at")
        .sort({ published_date: -1, created_at: -1 })
        .skip(skip)
        .limit(pageSize)
        .lean(),
      Blog.countDocuments(filter),
    ]);

    return NextResponse.json({
      success: true,
      data: data.map(toDTO),
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error("[GET /api/home/blogs] error:", error);
    return NextResponse.json(
      { success: false, message: "Internal server error" },
      { status: 500 }
    );
  }
}
