import dbConnect from "@/lib/dbConnect";
import LoginAttempt, {
  LOGIN_ATTEMPT_WINDOW_SECONDS,
} from "@/models/mongoose/LoginAttempt";
import logger from "@/utils/logger";

// Failed-attempt thresholds per key type. Both share the same 15-minute
// window as the LoginAttempt TTL (src/models/mongoose/LoginAttempt.ts).
const EMAIL_MAX_ATTEMPTS = 5;
const IP_MAX_ATTEMPTS = 20;
const WINDOW_MS = LOGIN_ATTEMPT_WINDOW_SECONDS * 1000;

export interface RateLimitCheck {
  limited: boolean;
}

function isWithinWindow(lastAttemptAt: Date): boolean {
  return Date.now() - lastAttemptAt.getTime() < WINDOW_MS;
}

/**
 * Extract the client IP the same way src/middleware.ts does: first entry of
 * x-forwarded-for, else x-real-ip. `request.ip` is not reliable on Cloud
 * Run TLS terminates at a proxy in front of the instance.
 */
export function getClientIp(request: Request): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) {
    const first = forwardedFor.split(",")[0]?.trim();
    if (first) return first;
  }
  return request.headers.get("x-real-ip") || "unknown";
}

/**
 * Check whether the given (already-normalized) email or IP is currently
 * locked out. Fails open: any store error is logged and treated as
 * "not limited" so a Mongo outage can never block logins outright.
 */
export async function checkLoginRateLimit(
  email: string,
  ip: string,
): Promise<RateLimitCheck> {
  try {
    await dbConnect();
    const docs = await LoginAttempt.find({
      $or: [
        { key: email, type: "email" },
        { key: ip, type: "ip" },
      ],
    }).lean();

    for (const doc of docs) {
      if (!isWithinWindow(doc.last_attempt_at)) continue;
      const max = doc.type === "email" ? EMAIL_MAX_ATTEMPTS : IP_MAX_ATTEMPTS;
      if (doc.attempts >= max) return { limited: true };
    }
    return { limited: false };
  } catch (error) {
    logger.error("Login rate-limit check failed, allowing attempt", {
      source: "loginRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
    return { limited: false };
  }
}

/**
 * Record a failed login attempt against both the email and IP keys.
 * Fails open a store failure here is logged and swallowed, never thrown
 * back into the caller.
 */
export async function recordFailedLoginAttempt(
  email: string,
  ip: string,
): Promise<void> {
  try {
    await dbConnect();
    const now = new Date();
    const keys: Array<{ key: string; type: "email" | "ip" }> = [
      { key: email, type: "email" },
      { key: ip, type: "ip" },
    ];
    await Promise.all(
      keys.map(({ key, type }) =>
        LoginAttempt.findOneAndUpdate(
          { key, type },
          {
            $inc: { attempts: 1 },
            $set: { last_attempt_at: now },
            $setOnInsert: { first_attempt_at: now },
          },
          { upsert: true },
        ),
      ),
    );
  } catch (error) {
    logger.error("Login rate-limit increment failed", {
      source: "loginRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Clear any tracked failed attempts for the email and IP after a
 * successful login. Fails open a store failure here just means stale
 * counters linger until their TTL expires, not a blocked login.
 */
export async function resetLoginAttempts(
  email: string,
  ip: string,
): Promise<void> {
  try {
    await dbConnect();
    await LoginAttempt.deleteMany({
      $or: [
        { key: email, type: "email" },
        { key: ip, type: "ip" },
      ],
    });
  } catch (error) {
    logger.error("Login rate-limit reset failed", {
      source: "loginRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
