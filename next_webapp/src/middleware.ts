import createMiddleware from "next-intl/middleware";
import { NextRequest, NextResponse } from "next/server";
import { locales } from "./i18n";


// next-intl middleware instance (handles locale detection + redirects)
const intlMiddleware = createMiddleware({
  locales,
  defaultLocale: "en",
});

// Route classification

/**
 * Page paths that require a valid JWT.
 * Matched against the path with any locale prefix stripped.
 */
const PROTECTED_PATHS = ["/dashboard", "/admin", "/upload", "/statistics", "/api/profile", "/api/categories"];

/**
 * Paths that are always publicly accessible and skip every auth check.
 * Matched against the bare path (locale prefix stripped).
 */
const PUBLIC_PATHS = new Set(["/", "/login", "/signup"]);

/**
 * API route prefixes that are always public (no auth token required).
 */
const PUBLIC_API_PREFIXES = ["/api/auth/login", "/api/auth/signup"];

// Strip a known locale prefix from the pathname, returning the bare path.
function getBarePath(pathname: string): string {
  for (const locale of locales) {
    const prefix = `/${locale}`;
    if (pathname === prefix) return "/";
    if (pathname.startsWith(`${prefix}/`)) return pathname.slice(prefix.length);
  }
  return pathname;
}

// Return true when the request path is a protected page or API route. 
function isProtectedPath(pathname: string): boolean {
  const bare = getBarePath(pathname);
  return PROTECTED_PATHS.some((p) => bare === p || bare.startsWith(`${p}/`));
}

// Return true when the request should bypass auth entirely.
function isPublicPath(pathname: string): boolean {
  // Always-public API prefixes (exact prefix match)
  if (PUBLIC_API_PREFIXES.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    return true;
  }
  // Always-public page paths (bare path match)
  const bare = getBarePath(pathname);
  return PUBLIC_PATHS.has(bare);
}

// Decode a base64url-encoded string into a Uint8Array.
function base64urlDecode(str: string): Uint8Array {
  const base64 = str.replace(/-/g, "+").replace(/_/g, "/");
  const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
  const binary = atob(padded);
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

/**
 * Verify an HS256 JWT and return its decoded payload, or null on any failure
 * (bad structure, invalid signature, expired token).
 */
async function verifyJWT(
  token: string,
  secret: string,
): Promise<Record<string, unknown> | null> {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;

    const [headerB64, payloadB64, signatureB64] = parts;

    // Import the HMAC-SHA256 secret key
    const keyData = new TextEncoder().encode(secret);
    const cryptoKey = await crypto.subtle.importKey(
      "raw",
      keyData,
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["verify"],
    );

    // Verify the signature over "header.payload"
    const signingInput = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
    const signature = base64urlDecode(signatureB64);
    const isValid = await crypto.subtle.verify(
      "HMAC",
      cryptoKey,
      signature,
      signingInput,
    );
    if (!isValid) return null;

    // Decode and parse the payload
    const payloadJson = new TextDecoder().decode(base64urlDecode(payloadB64));
    const payload = JSON.parse(payloadJson) as Record<string, unknown>;

    // Reject expired tokens
    if (typeof payload.exp === "number" && payload.exp < Date.now() / 1000) {
      return null;
    }

    return payload;
  } catch {
    return null;
  }
}

// Middleware entry point

export default async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 1. Always-public paths: skip auth and delegate locale routing to intl
  if (isPublicPath(pathname)) {
    if (!pathname.startsWith("/api/")) {
      return intlMiddleware(request);
    }
    return NextResponse.next();
  }

  // 2. Non-protected paths: no auth needed, intl handles page routes
  if (!isProtectedPath(pathname)) {
    if (!pathname.startsWith("/api/")) {
      return intlMiddleware(request);
    }
    return NextResponse.next();
  }

  // 3. Page routes: browsers don't send Authorization headers on navigation.
  //    Skip JWT check here — protection is handled client-side by the admin
  //    layout guard (checks localStorage for token + role).
  if (!pathname.startsWith("/api/")) {
    return intlMiddleware(request);
  }

  // 4. Protected API route: verify the JWT
  const JWT_SECRET = process.env.JWT_SECRET;
  if (!JWT_SECRET) {
    // Misconfigured server — fail closed
    return NextResponse.json(
      { success: false, message: "Server configuration error" },
      { status: 500 },
    );
  }

  // Extract token from "Authorization: Bearer <token>" header
  const authHeader = request.headers.get("authorization");
  const token =
    authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;

  if (!token) {
    return NextResponse.json(
      { success: false, message: "Unauthorised" },
      { status: 401 },
    );
  }

  const payload = await verifyJWT(token, JWT_SECRET);
  if (!payload) {
    return NextResponse.json(
      { success: false, message: "Unauthorised" },
      { status: 401 },
    );
  }

  // 5. Token is valid: attach decoded claims to request headers for
  //    downstream route handlers and server components.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-user-id", String(payload.userId ?? ""));
  requestHeaders.set("x-user-role", String(payload.roleName ?? ""));
  requestHeaders.set("x-user-role-id", String(payload.roleId ?? ""));

  // Protected API route: forward with modified headers only.
  return NextResponse.next({ request: { headers: requestHeaders } });
}

export const config = {
  matcher: [

    // next-intl required patterns
    "/",
    "/(cn|en|es|el|ar|it|hi|vi)/:path*",
    // Protected page routes
    "/dashboard/:path*",
    "/admin/:path*",
    "/upload/:path*",
    "/statistics/:path*",
    
    // Protected API routes — profile
    "/api/profile",
    "/api/profile/:path*",

    // Protected API routes — category
    "/api/categories",          
    "/api/categories/:path*",

    // Public auth API routes (handled by isPublicPath — pass straight through)
    "/api/auth/:path*",
  ],
};