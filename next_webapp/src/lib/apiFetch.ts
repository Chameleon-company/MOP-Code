import { toast } from "sonner";

/**
 * Thrown by apiFetch for any failed request: network failure, non-2xx
 * HTTP status, or a 2xx response whose JSON body has `success: false`.
 *
 * `status` is 0 for network-level failures (offline, DNS, CORS, server
 * unreachable) where no HTTP response was ever received.
 */
export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

interface ApiFetchOptions extends RequestInit {
  /**
   * Suppress the automatic error toast for this call. Use when the
   * caller wants to render its own inline error UI instead (e.g. a
   * form field) rather than a global notification. The call still
   * throws ApiError on failure either way.
   */
  silent?: boolean;
}

/**
 * fetch() wrapper that centralizes error handling for API calls:
 *  - Network failures (server down, offline, CORS, timeouts) are caught
 *    and normalized instead of throwing a raw TypeError.
 *  - Non-2xx HTTP responses are treated as failures.
 *  - This app's API routes return `{ success, message, data }`; a 2xx
 *    response with `success: false` is also treated as a failure.
 *  - On any failure, a toast is shown (unless `silent`) and an
 *    `ApiError` is thrown so callers can still `catch` to update local
 *    state (loading flags, fallback UI, etc.) without re-implementing
 *    error messaging themselves.
 *
 * On success, resolves with the parsed JSON body (the full response
 * envelope - callers destructure `.data` as needed, matching this
 * app's existing `{ success, data }` API shape).
 */
export async function apiFetch<T = unknown>(
  input: string,
  options: ApiFetchOptions = {}
): Promise<T> {
  const { silent, ...init } = options;

  let res: Response;
  try {
    res = await fetch(input, init);
  } catch {
    const message = "Network error - please check your connection and try again.";
    if (!silent) toast.error(message);
    throw new ApiError(message, 0);
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // No/invalid JSON body. Fine for e.g. 204 No Content; only a
    // problem below if the response also wasn't ok.
  }

  const success =
    typeof body === "object" && body !== null && "success" in body
      ? (body as { success?: unknown }).success !== false
      : true;

  if (!res.ok || !success) {
    const message =
      (typeof body === "object" && body !== null && "message" in body
        ? String((body as { message?: unknown }).message)
        : undefined) || `Request failed (${res.status})`;
    if (!silent) toast.error(message);
    throw new ApiError(message, res.status, body);
  }

  return body as T;
}
