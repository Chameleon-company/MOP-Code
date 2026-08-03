import { NextRequest, NextResponse } from "next/server";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import User from "@/models/mongoose/User";
import { errorResponse } from "@/app/api/library/errorResponse";

export async function GET(request: NextRequest) {
  try {
    await dbConnect();

    // The middleware verifies the JWT and adds this header.
    const userId = request.headers.get("x-user-id");

    if (!userId || !mongoose.Types.ObjectId.isValid(userId)) {
      return errorResponse(
        "Unauthorised",
        401,
        "UNAUTHORIZED",
        request,
      );
    }

    // Fetch the current session user from MongoDB.
    const user = await User.findById(userId).exec();

    if (!user) {
      return errorResponse(
        "Unauthorised",
        401,
        "UNAUTHORIZED",
        request,
      );
    }

    const roleId = user.role?.legacy_id
      ? Number(user.role.legacy_id)
      : null;

    return NextResponse.json(
      {
        success: true,
        message: "Session retrieved successfully",
        data: {
          userId: String(user._id),
          email: user.email,
          firstName: user.profile?.first_name ?? null,
          lastName: user.profile?.last_name ?? null,
          age: user.profile?.age ?? null,
          gender: user.profile?.gender ?? null,
          profileImg: user.profile?.profile_img ?? null,
          roleId,
          roleName: user.role?.role_name ?? null,
        },
      },
      { status: 200 },
    );
  } catch (error) {
    console.error("Session Error:", error);

    return errorResponse(
      "Internal Server Error",
      500,
      "INTERNAL_ERROR",
      request,
    );
  }
}