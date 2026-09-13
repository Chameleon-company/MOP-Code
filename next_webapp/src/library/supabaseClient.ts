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

// TODO: remove once blogs/categories/gallery move to MongoDB ( will get a pr from this and remov in next brabanch )
// Lazy proxy — defers client creation to first property access so the
// module can be imported at build time without env vars present.
export const supabase = new Proxy({} as SupabaseClient, {
  get(_target, prop, receiver) {
    const client = getSupabase();
    const value = Reflect.get(client, prop, client);
    return typeof value === 'function' ? value.bind(client) : value;
  },
});