import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

export async function GET(request) {
  try {
    const { data, error } = await supabase
      .from('usecases')
      .select('*');

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json(
        { success: false, error: 'Failed to fetch use cases' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      data: data || []
    });
  } catch (error) {
    console.error('Error fetching use cases:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch use cases' },
      { status: 500 }
    );
  }
}
