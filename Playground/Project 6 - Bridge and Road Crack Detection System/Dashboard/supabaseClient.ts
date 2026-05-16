import { createClient, SupabaseClient } from '@supabase/supabase-js'


const supabaseUrl = (import.meta as any).env.VITE_SUPABASE_URL
const supabaseKey = (import.meta as any).env.VITE_SUPABASE_PUBLISHABLE_KEY

function createDatabaseClients() {
    if (!supabaseUrl || !supabaseKey){
        console.error('Missing Supabase environment variables')
        return {supabase: null}
    }

    let supabase: SupabaseClient

    try {
        supabase = createClient(supabaseUrl, supabaseKey)
    }
    catch (error) {
        console.error('Failed to create supabase client', error)
        return {supabase: null}
    }

    return {supabase}



}

const {supabase} = createDatabaseClients()

export {supabase}
    

