// import { createClient } from '@supabase/supabase-js';

// const supabaseUrl = process.env.SUPABASE_URL!;
// const supabaseKey = process.env.SUPABASE_API_KEY!;

// export const supabase = createClient(supabaseUrl, supabaseKey);
import { createClient, SupabaseClient } from '@supabase/supabase-js';

let _supabase: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (_supabase) return _supabase;

  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_API_KEY;

  if (!supabaseUrl || !supabaseKey) {
    throw new Error('Supabase env vars are not configured.');
  }

  _supabase = createClient(supabaseUrl, supabaseKey);
  return _supabase;
}