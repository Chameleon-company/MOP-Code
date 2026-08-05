import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect'; 
import UseCaseModel from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';
export const runtime = 'nodejs'

// GET /api/statistics/total-count
// Returns the total number of use cases.
// Public — no auth required.
export async function GET() {
  try {
    await dbConnect();
    
    const count = await UseCaseModel.countDocuments();

    return NextResponse.json({ success: true, total: count });
  } catch (error) {
    console.error('[GET /api/statistics/total-count] unexpected error:', error);
    return errorResponse('Internal server error', 500);
  }
}