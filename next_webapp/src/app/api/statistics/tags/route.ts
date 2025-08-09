// app/api/statistics/tags/route.ts
import { NextRequest, NextResponse } from 'next/server';
import dbConnect from '../../../../../lib/postgresql';

export async function GET(req: NextRequest) {
  try {
    const db = await dbConnect();
    const useCaseCollection = db.collection('usecases');
    
    const { searchParams } = new URL(req.url);
    const search = searchParams.get('search') || '';
    const page = parseInt(searchParams.get('page') || '1', 10);
    const limit = parseInt(searchParams.get('limit') || '5', 10);
    
    // Debug logging
    console.log("Search query:", search);
    
    // Get all tags first
    const tagStats = await useCaseCollection.aggregate([
      {
        $group: {
          _id: "$tag",
          testCasesPublished: { $sum: 1 }
        }
      },
      {
        $project: {
          tag: "$_id",
          testCasesPublished: 1,
          _id: 0
        }
      },
      { $sort: { testCasesPublished: -1 } }
    ]).toArray();
    
    console.log("Total tags found:", tagStats.length);
    
    // Apply search filter case-insensitively
    const filtered = search 
      ? tagStats.filter((item: { tag: string }) => {
          // Case insensitive search
          const normalizedTag = item.tag.toLowerCase();
          const normalizedSearch = search.toLowerCase();
          const result = normalizedTag.includes(normalizedSearch);
          
          console.log(`Checking tag "${item.tag}" against "${search}": ${result}`);
          return result;
        })
      : tagStats;
    
    console.log("Filtered tags:", filtered.length);
    
    const total = filtered.length;
    
    // Pagination based on filtered data
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    
    console.log(`Pagination: page ${page}, limit ${limit}, range ${startIndex}-${endIndex}`);
    
    // Ensure we do not exceed available filtered results
    const paginated = filtered.slice(startIndex, endIndex);
    
    console.log("Paginated results:", paginated.length);
    
    const totalTestCases = filtered.reduce(
      (sum: number, item: { testCasesPublished: number }) => 
        sum + item.testCasesPublished, 0
    );
    
    const result = paginated.map((item: { tag: string; testCasesPublished: number }, index: number) => ({
      no: startIndex + index + 1,
      tag: item.tag,
      numberOfTestCasesPublished: item.testCasesPublished,
      popularity: `${((item.testCasesPublished / totalTestCases) * 100).toFixed(2)}%`
    }));
    
    return NextResponse.json({
      total,
      data: result,
      pagination: {
        currentPage: page,
        totalPages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    console.error('Failed to fetch tag stats:', error);
    return NextResponse.json({ error: 'Failed to fetch tag stats' }, { status: 500 });
  }
}
