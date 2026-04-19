import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

export async function GET(request, { params }) {
  try {
    const { id } = params;

    // Validate id is a number
    if (isNaN(id)) {
      return NextResponse.json(
        { success: false, error: 'Invalid use case ID' },
        { status: 400 }
      );
    }

    const { data, error } = await supabase
      .from('usecases')
      .select('*')
      .eq('id', parseInt(id))
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return NextResponse.json(
          { success: false, error: 'Use case not found' },
          { status: 404 }
        );
      }
      console.error('Supabase error:', error);
      return NextResponse.json(
        { success: false, error: 'Failed to fetch use case' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Error fetching use case:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch use case' },
      { status: 500 }
    );
  }
}
