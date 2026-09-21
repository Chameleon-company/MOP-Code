// Non-secret configuration for public-page audits; never connect CI to production data.
const { spawnSync } = require('node:child_process');
const env = {
  ...process.env,
  NEXT_TELEMETRY_DISABLED: '1',
  SUPABASE_URL: 'http://127.0.0.1:54321',
  SUPABASE_API_KEY: 'lighthouse-ci-placeholder',
  DATABASE_URL: 'postgresql://lighthouse:lighthouse@127.0.0.1:5432/lighthouse',
  MONGODB_URI: 'mongodb://127.0.0.1:27017/lighthouse-ci?serverSelectionTimeoutMS=1000',
  JWT_SECRET: 'lighthouse-ci-local-only-not-a-production-secret',
  NEXT_PUBLIC_FIREBASE_API_KEY: 'lighthouse-ci-placeholder',
  NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN: 'lighthouse-ci.invalid',
  NEXT_PUBLIC_FIREBASE_PROJECT_ID: 'lighthouse-ci',
  NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET: 'lighthouse-ci.invalid',
  NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID: '1234567890',
  NEXT_PUBLIC_FIREBASE_APP_ID: '1:1234567890:web:lighthouseci',
  NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID: '',
  SENTRY_DSN: '',
  NEXT_PUBLIC_SENTRY_DSN: '',
};
const [command, ...args] = process.argv.slice(2);
if (!command) throw new Error('Usage: node scripts/lighthouse-env.cjs <command> [args...]');
const result = spawnSync(command, args, { env, stdio: 'inherit', shell: process.platform === 'win32' });
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
