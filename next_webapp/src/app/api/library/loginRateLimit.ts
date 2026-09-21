import dbConnect from "@/lib/dbConnect";
import LoginAttempt, {
  LOGIN_ATTEMPT_WINDOW_SECONDS,
} from "@/models/mongoose/LoginAttempt";
import logger from "@/utils/logger";
import * as Sentry from "@sentry/nextjs";

// Failed-attempt thresholds per key type. Both share the same 15-minute
// window as the LoginAttempt TTL (src/models/mongoose/LoginAttempt.ts).
const EMAIL_MAX_ATTEMPTS = 5;

// Kept deliberately loose relative to EMAIL_MAX_ATTEMPTS: the per-IP limit
// only needs to catch real floods (credential stuffing = thousands of
// attempts), not fine-grained abuse that's the email key's job. And
// because checkLoginRateLimit increments this counter speculatively for
// EVERY request, before auth resolves (see below), a burst of legitimate
// simultaneous logins from one shared NAT/VPN IP (e.g. a university class
// logging in at once) can transiently push the count up even though none
// of them are attacks. 100 comfortably absorbs a class/lab-sized burst
// while still shutting down an actual flood.
const IP_MAX_ATTEMPTS = 100;
const WINDOW_MS = LOGIN_ATTEMPT_WINDOW_SECONDS * 1000;
const EPOCH = new Date(0);

export interface RateLimitCheck {
  limited: boolean;
  // Which key actually pushed this request over its limit. Only set when
  // `limited` is true. The IP key is checked first and short-circuits
  // before the email key is ever touched (see checkLoginRateLimit), so a
  // request is always attributable to exactly one key here.
  limitedBy?: "email" | "ip";
}

/**
 * Extract the client IP from x-forwarded-for, else x-real-ip. `request.ip`
 * is not reliable on Cloud Run TLS terminates at a proxy in front of the
 * instance.
 *
 * Takes the LAST entry of x-forwarded-for, not the first: Cloud Run appends
 * the real client IP to the end of the chain, while every entry before that
 * is attacker-controlled request-header content. Trusting the first entry
 * lets a client spoof any IP and bypass the per-IP limit entirely.
 *
 * Trusting the last entry unconditionally (no verification that the
 * request actually arrived through a trusted proxy) is working ONLY because
 * of this app's current deployment topology: it is deployed as a
 * fully-managed Cloud Run service (see jenkinsfile, `gcloud run deploy
 * ... --platform managed`), where ALL traffic terminates at Google's front
 * end / Cloud Run's HTTPS proxy a client has no way to reach the
 * container directly and inject a fabricated entry after their own.
 *
 * This assumption breaks, and last-entry trust would need to become real
 * trusted-proxy validation (e.g. checking the immediate peer against
 * Google's published front-end IP ranges), if this service is ever put
 * behind an additional CDN/reverse proxy that forwards x-forwarded-for
 * verbatim, or if the networking model changes so the container becomes
 * reachable other than via Cloud Run's managed front end.
 */
export function getClientIp(request: Request): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) {
    const parts = forwardedFor
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean);
    if (parts.length > 0) return parts[parts.length - 1];
  }
  return request.headers.get("x-real-ip") || "unknown";
}

/**
 * Atomically increments the attempt counter for a single (key, type) pair
 * and returns the resulting count. This is the same window-aware pipeline
 * as before: resets to 1 when the previous attempt fell outside WINDOW_MS,
 * otherwise increments, and always bumps last_attempt_at (which is what
 * gives the sliding lockout window see the TTL index comment in
 * LoginAttempt.ts).
 *
 * Callers MUST derive their limit decision from this call's own returned
 * count, never from an earlier separate read: a single findOneAndUpdate is
 * atomic, so concurrent callers racing on the same key are serialized by
 * Mongo and each gets back a count that reflects every write that landed
 * before it, not a stale snapshot.
 */
async function incrementAttempt(
  key: string,
  type: "email" | "ip",
  now: Date,
): Promise<number> {
  const doc = await LoginAttempt.findOneAndUpdate(
    { key, type },
    [
      {
        $set: {
          _stale: {
            $gte: [
              { $subtract: [now, { $ifNull: ["$last_attempt_at", EPOCH] }] },
              WINDOW_MS,
            ],
          },
        },
      },
      {
        $set: {
          attempts: {
            $cond: [
              "$_stale",
              1,
              { $add: [{ $ifNull: ["$attempts", 0] }, 1] },
            ],
          },
          first_attempt_at: {
            $cond: ["$_stale", now, { $ifNull: ["$first_attempt_at", now] }],
          },
          last_attempt_at: now,
        },
      },
      { $unset: "_stale" },
    ],
    { upsert: true, new: true },
  ).lean();

  if (!doc) {
    // Not reachable in practice (upsert: true, new: true always returns
    // the post-update document) — guards the type and, if it ever did
    // happen, routes into the caller's fail-open catch instead of a crash.
    throw new Error(`Rate-limit upsert returned no document for ${type} key`);
  }
  return doc.attempts;
}

/**
 * Check whether the given (already-normalized) email or IP is currently
 * locked out, atomically recording this attempt in the same step (see
 * incrementAttempt) so the limit decision is never made from a value a
 * concurrent request could have already changed. Fails open: any store
 * error is logged and treated as "not limited" so a Mongo outage can never
 * block logins outright.
 *
 * A successful login's resetLoginAttempts call afterward erases whatever
 * this function just recorded, so a request with valid credentials never
 * leaves a trace here even though it was speculatively counted on the way
 * in.
 *
 * The IP key is checked (and bumped) FIRST. If it alone is already over
 * its limit, the request is rejected immediately WITHOUT ever touching the
 * email key an already-IP-limited attacker rotating through novel,
 * never-seen emails does not get a fresh login_attempts document created
 * per request just to tell us what we already know from the IP.
 */
export async function checkLoginRateLimit(
  email: string,
  ip: string,
): Promise<RateLimitCheck> {
  try {
    await dbConnect();
    const now = new Date();

    const ipAttempts = await incrementAttempt(ip, "ip", now);
    if (ipAttempts > IP_MAX_ATTEMPTS) {
      return { limited: true, limitedBy: "ip" };
    }

    const emailAttempts = await incrementAttempt(email, "email", now);
    if (emailAttempts > EMAIL_MAX_ATTEMPTS) {
      return { limited: true, limitedBy: "email" };
    }

    return { limited: false };
  } catch (error) {
    logger.error("Login rate-limit check failed, allowing attempt", {
      source: "loginRateLimit",
      code: "RATE_LIMIT_FAIL_OPEN",
      error: error instanceof Error ? error.message : String(error),
    });
    Sentry.captureMessage(
      "Login rate-limit store error - failing open, rate limiting is OFF",
      {
        level: "error",
        tags: { source: "loginRateLimit", code: "RATE_LIMIT_FAIL_OPEN" },
      },
    );
    return { limited: false };
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
