import winston from 'winston';
import { Types } from 'mongoose';
import dbConnect from '@/lib/dbConnect';
import Log from '@/models/mongoose/Log';
import User from '@/models/mongoose/User';

interface LogEntry {
  level: string;
  message: string;
  timestamp: Date;
  meta?: Record<string, unknown>;
  source?: string;
  user_id?: number | string | null;
  ip_address?: string | null;
  user_agent?: string;
  method?: string;
  url?: string;
  status_code?: number;
  response_time?: number;
}

const KNOWN_FIELDS = new Set<string | symbol>([
  'level',
  'message',
  'source',
  'user_id',
  'ip_address',
  'user_agent',
  'method',
  'url',
  'status_code',
  'response_time',
  'timestamp',
  'splat',
  'service',
  Symbol.for('level'),
  Symbol.for('splat'),
]);

const FLUSH_INTERVAL_MS = 10_000;
const FLUSH_BATCH_SIZE = 20;

// Removes Winston colour codes before logs are written to MongoDB.
const ANSI_ESCAPE_PATTERN =
  /\u001B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;

function removeAnsi(value: unknown): string {
  return String(value ?? '').replace(ANSI_ESCAPE_PATTERN, '');
}

class DatabaseTransport extends winston.Transport {
  private buffer: LogEntry[] = [];
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  private isFlushing = false;

  constructor(opts?: any) {
    super(opts);

    this.flushTimer = setInterval(() => {
      void this.flush();
    }, FLUSH_INTERVAL_MS);

    if (this.flushTimer.unref) {
      this.flushTimer.unref();
    }
  }

  log(info: any, callback: () => void) {
    const meta: Record<string, unknown> = {};

    for (const key of Object.keys(info)) {
      if (!KNOWN_FIELDS.has(key)) {
        meta[key] = info[key];
      }
    }

    const rawIp =
      typeof info.ip_address === 'string' ? info.ip_address : undefined;

    const firstIp = rawIp?.split(',')[0].trim();

    const validIp =
      firstIp && /^[\d.:a-fA-F]+$/.test(firstIp) ? firstIp : null;

    this.buffer.push({
      level: removeAnsi(info.level),
      message: removeAnsi(info.message),
      timestamp: new Date(),
      meta: Object.keys(meta).length ? meta : undefined,
      source: info.source || 'application',
      user_id: info.user_id ?? null,
      ip_address: validIp,
      user_agent: info.user_agent,
      method: info.method,
      url: info.url,
      status_code: info.status_code,
      response_time: info.response_time,
    });

    if (this.buffer.length >= FLUSH_BATCH_SIZE) {
      void this.flush();
    }

    callback();
  }

  async flush(): Promise<void> {
    if (this.isFlushing || this.buffer.length === 0) {
      return;
    }

    this.isFlushing = true;
    const batch = this.buffer.splice(0, this.buffer.length);

    try {
      await dbConnect();

      const legacyUserIds = Array.from(
        new Set(
          batch
            .map((entry) => entry.user_id)
            .filter(
              (userId): userId is number | string =>
                userId !== null && userId !== undefined,
            )
            .map(String)
            .filter((userId) => !Types.ObjectId.isValid(userId)),
        ),
      );

      const userIdMap = new Map<string, unknown>();

      if (legacyUserIds.length > 0) {
        const users = await User.find({
          legacy_id: { $in: legacyUserIds },
        })
          .select('_id legacy_id')
          .lean();

        for (const user of users) {
          if (user.legacy_id) {
            userIdMap.set(user.legacy_id, user._id);
          }
        }
      }

      const documents = batch.map((entry) => {
        const { user_id: userId, ...logData } = entry;
        const userIdString =
          userId !== null && userId !== undefined ? String(userId) : null;

        let mongoUserId: unknown = null;

        if (userIdString) {
          mongoUserId = Types.ObjectId.isValid(userIdString)
            ? userIdString
            : userIdMap.get(userIdString) ?? null;
        }

        return {
          ...logData,
          user_id: mongoUserId,
        };
      });

      await Log.insertMany(documents, { ordered: false });
    } catch (error) {
      console.error('Database transport flush error:', error);
    } finally {
      this.isFlushing = false;
    }
  }

  close() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }

    void this.flush();
  }
}

export default DatabaseTransport;