import winston from 'winston';
import { supabase } from '@/library/supabaseClient';

interface LogEntry {
  level: string;
  message: string;
  timestamp: string;
  meta?: any;
  source?: string;
  user_id?: number;
  ip_address?: string | null;
  user_agent?: string;
  method?: string;
  url?: string;
  status_code?: number;
  response_time?: number;
}

const KNOWN_FIELDS = new Set<string | symbol>([
  'level', 'message', 'source', 'user_id', 'ip_address', 'user_agent',
  'method', 'url', 'status_code', 'response_time',
  'timestamp', 'splat', 'service', Symbol.for('level'), Symbol.for('splat'),
]);

const FLUSH_INTERVAL_MS = 10_000; // flush every 10 seconds
const FLUSH_BATCH_SIZE = 20;      // flush early if buffer reaches 20 entries

class DatabaseTransport extends winston.Transport {
  private buffer: LogEntry[] = [];
  private flushTimer: ReturnType<typeof setInterval> | null = null;

  constructor(opts?: any) {
    super(opts);
    this.flushTimer = setInterval(() => this.flush(), FLUSH_INTERVAL_MS);
    // Allow the process to exit without waiting for the timer
    if (this.flushTimer.unref) this.flushTimer.unref();
  }

  log(info: any, callback: () => void) {
    const meta: Record<string, unknown> = {};
    for (const key of Object.keys(info)) {
      if (!KNOWN_FIELDS.has(key)) {
        meta[key] = (info as Record<string, unknown>)[key];
      }
    }

    const rawIp: string | undefined = info.ip_address;
    const validIp = rawIp && /^[\d.:a-fA-F]+$/.test(rawIp.split(',')[0].trim())
      ? rawIp.split(',')[0].trim()
      : null;

    this.buffer.push({
      level: info.level,
      message: info.message,
      timestamp: new Date().toISOString(),
      meta: Object.keys(meta).length ? meta : undefined,
      source: info.source || 'application',
      user_id: info.user_id,
      ip_address: validIp,
      user_agent: info.user_agent,
      method: info.method,
      url: info.url,
      status_code: info.status_code,
      response_time: info.response_time,
    });

    if (this.buffer.length >= FLUSH_BATCH_SIZE) {
      this.flush();
    }

    callback();
  }

  private async flush() {
    if (this.buffer.length === 0) return;
    const batch = this.buffer.splice(0, this.buffer.length);
    try {
      const { error } = await supabase.from('logs').insert(batch);
      if (error) console.error('Failed to insert log batch into database:', error);
    } catch (error) {
      console.error('Database transport flush error:', error);
    }
  }
}

export default DatabaseTransport;
