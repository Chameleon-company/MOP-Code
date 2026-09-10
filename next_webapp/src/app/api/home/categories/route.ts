import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import Category from '@/models/mongoose/Category';
import { errorResponse } from '@/app/api/library/errorResponse';

// GET /api/home/categories
// Returns all categories with name, description and cover image.
// Public — no auth required (used on the home page).
export async function GET() {
  try {
    await dbConnect();

    const data = await Category.find({})
      .select('category_name description cover_img')
      .sort({ category_name: 1 })
      .lean();

    return NextResponse.json({
      success: true,
      data: data.map(({ _id, __v, ...rest }: any) => ({ id: _id.toString(), ...rest })),
    });
  } catch (error) {
    console.error('[GET /api/home/categories] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
