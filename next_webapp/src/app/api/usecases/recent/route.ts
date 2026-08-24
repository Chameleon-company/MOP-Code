import { NextResponse } from 'next/server';
import dbConnect from '@/lib/dbConnect';
import UseCase from '@/models/mongoose/UseCase';
import { errorResponse } from '@/app/api/library/errorResponse';
import { toUseCaseDTO } from '@/app/api/library/useCaseDto';

// GET /api/usecases/recent
// Returns the 4 most recently created use cases (Mongo-backed — tags are
// embedded on the doc already, see TagRefSchema in UseCase.ts, so no join
// lookup is needed here).
// Public — no auth required (used on the home page).
export async function GET() {
  try {
    await dbConnect();

    const docs = await UseCase.find({})
      .sort({ created_at: -1 })
      .limit(4)
      .lean();

    return NextResponse.json({ success: true, data: docs.map(toUseCaseDTO) });
  } catch (error) {
    console.error('[GET /api/usecases/recent] unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
}
