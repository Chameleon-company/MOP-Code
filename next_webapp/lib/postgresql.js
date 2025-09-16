const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_API_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing SUPABASE_URL or SUPABASE_API_KEY environment variables');
}

const supabase = createClient(supabaseUrl, supabaseKey);
module.exports = supabase;

async function testConnection() {
  const { data, error } = await supabase.from('users').select('*').limit(1);
  if (error) {
    console.error('Error connecting to Supabase:', error);
    process.exit(1);
  }
  console.log('Supabase connection successful. Sample data:', data);
}

testConnection();

// // lib/mongodb.js
// const mongoose = require('mongoose');

// const MONGODB_URI = process.env.MONGODB_URI;

// if (!MONGODB_URI) {
//   throw new Error('Please define the MONGODB_URI environment variable inside .env.local');
// }

// let cached = global.mongoose;

// if (!cached) {
//   cached = global.mongoose = { conn: null, promise: null };
// }

// async function dbConnect() {
//   if (cached.conn) {
//     console.log('Using existing connection');
//     return cached.conn;
//   }

//   if (!cached.promise) {
//     const opts = {
//       bufferCommands: false,
//     };

//     cached.promise = mongoose.connect(MONGODB_URI, opts).then((mongoose) => {
//       console.log('Connected to MongoDB');
//       return mongoose.connection;
//     });
//   }
//   cached.conn = await cached.promise;
//   return cached.conn;
// }

// module.exports = dbConnect;

