// app/api/statistics/trimester/route.ts
import { NextResponse } from 'next/server';
import dbConnect from '../../../../../lib/postgresql';

export async function GET() {
  try {
    const db = await dbConnect();
    const useCaseCollection = db.collection('usecases');

    const trimesterStats = await useCaseCollection.aggregate([
      {
        $project: {
          trimester: { $substr: ['$semester', 5, 2] } 
        }
      },
      {
        $group: {
          _id: '$trimester',
          count: { $sum: 1 }
        }
      },
      {
        $sort: { _id: 1 }
      }
    ]).toArray();

    const formattedTrimesterStats = {
      labels: trimesterStats.map((item: { _id: string }) => `Trimester ${item._id.replace('T', '')}`),
      data: trimesterStats.map((item: { count: number }) => item.count)
    };

    return NextResponse.json(formattedTrimesterStats);

  } catch (error) {
    console.error('Error generating trimester stats:', error);
    return NextResponse.json(
      { error: 'Unable to generate trimester stats' },
      { status: 500 }
    );
  }
}
