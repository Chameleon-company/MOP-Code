import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect'; 
import UseCaseModel from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';
export const runtime = 'nodejs'

// GET /api/statistics/by-category
// Returns use case counts grouped by category.
// Public — no auth required.
export async function GET() {
  try {
    await dbConnect();
    
    // MongoDB aggregation to group usecases by category and count them
    const result = await UseCaseModel.aggregate([
      {
        $group: {
          _id: '$category.category_name',
          count: { $sum: 1 }
        },
      }, 
      {
        $project: {
          _id: 0,
          category: {
            $ifNull: ['$_id', 'Uncategorized']
          },
          count: 1
        }
      }, 
      {
        $sort: { 
          count: -1 
        } 
      }
    ]);

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    console.error('[GET /api/statistics/by-category] unexpected error:', error);
    return errorResponse('Internal server error', 500);
  }
}