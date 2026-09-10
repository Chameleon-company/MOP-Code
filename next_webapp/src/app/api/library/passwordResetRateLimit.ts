import dbConnect from "@/lib/dbConnect";
import PasswordResetRateLimit, {
  PASSWORD_RESET_WINDOW_SECONDS,
} from "@/models/mongoose/PasswordResetRateLimit";
import logger from "@/utils/logger";

const FORGOT_PASSWORD_MAX_ATTEMPTS = 3;
const RESET_PASSWORD_MAX_ATTEMPTS = 5;
const WINDOW_MS = PASSWORD_RESET_WINDOW_SECONDS * 1000;

export interface RateLimitCheck {
  limited: boolean;
}

function isWithinWindow(lastAttemptAt: Date): boolean {
  return Date.now() - lastAttemptAt.getTime() < WINDOW_MS;
}

export function getClientIp(request: Request): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) {
    const ips = forwardedFor.split(",");
    const last = ips[ips.length - 1]?.trim();
    if (last) return last;
  }
  return request.headers.get("x-real-ip") || "unknown";
}

export async function checkPasswordResetRateLimit(
  email: string,
  ip: string,
  action: "forgot_password_request" | "failed_reset_attempt",
): Promise<RateLimitCheck> {
  try {
    await dbConnect();
    const docs = await PasswordResetRateLimit.find({
      $or: [
        { key: email, type: "email", action },
        { key: ip, type: "ip", action },
      ],
    }).lean();

    const maxAttempts = action === "forgot_password_request" ? FORGOT_PASSWORD_MAX_ATTEMPTS : RESET_PASSWORD_MAX_ATTEMPTS;

    for (const doc of docs) {
      if (!isWithinWindow(doc.last_attempt_at)) continue;
      if (doc.attempts >= maxAttempts) return { limited: true };
    }
    return { limited: false };
  } catch (error) {
    logger.error("Password reset rate-limit check failed, allowing attempt", {
      source: "passwordResetRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
    return { limited: false };
  }
}

export async function recordPasswordResetAttempt(
  email: string,
  ip: string,
  action: "forgot_password_request" | "failed_reset_attempt",
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
        PasswordResetRateLimit.findOneAndUpdate(
          { key, type, action },
          [
            {
              $set: {
                attempts: {
                  $cond: {
                    if: { $lt: ["$last_attempt_at", new Date(now.getTime() - WINDOW_MS)] },
                    then: 1,
                    else: { $add: [{ $ifNull: ["$attempts", 0] }, 1] }
                  }
                },
                first_attempt_at: {
                  $cond: {
                    if: { $lt: ["$last_attempt_at", new Date(now.getTime() - WINDOW_MS)] },
                    then: now,
                    else: { $ifNull: ["$first_attempt_at", now] }
                  }
                },
                last_attempt_at: now
              }
            }
          ],
          { upsert: true }
        ),
      ),
    );
  } catch (error) {
    logger.error("Password reset rate-limit increment failed", {
      source: "passwordResetRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

export async function clearPasswordResetAttempts(
  email: string,
  ip: string,
  action: "forgot_password_request" | "failed_reset_attempt",
): Promise<void> {
  try {
    await dbConnect();
    await PasswordResetRateLimit.deleteMany({
      $or: [
        { key: email, type: "email", action },
        { key: ip, type: "ip", action },
      ],
    });
  } catch (error) {
    logger.error("Password reset rate-limit reset failed", {
      source: "passwordResetRateLimit",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
