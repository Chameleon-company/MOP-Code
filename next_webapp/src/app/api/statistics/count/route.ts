import { NextRequest, NextResponse } from 'next/server';
import dbConnect from '../../../../../lib/postgresql';

export async function GET(req: NextRequest) {
  try {
  
    const db = await dbConnect(); 
    const useCaseCollection = db.collection('usecases'); 

    const { searchParams } = new URL(req.url);
    const trimester = searchParams.get('trimester'); 
    const tag = searchParams.get('tag');            
    const filter: any = {};

    if (trimester && trimester !== 'All') {
      filter.semester = trimester; 
    }

    if (tag && tag !== 'All') {
      filter.tag = tag;
    }


    const totalCount = await useCaseCollection.countDocuments(filter);

    return NextResponse.json({ total: totalCount });

  } catch (error) {
    console.error('Error fetching filtered count:', error);
    return NextResponse.json({ error: 'Failed to fetch count' }, { status: 500 });
  }
}
