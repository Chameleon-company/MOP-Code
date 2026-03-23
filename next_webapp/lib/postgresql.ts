import { Pool } from 'pg';

if (!process.env.DATABASE_URL) {
    throw new Error(
        'Missing required environment variable: DATABASE_URL\n' +
        'Copy .env.example to .env and fill in your PostgreSQL connection string.'
    );
}

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
});

export default pool;
