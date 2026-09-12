import type { MetadataRoute } from "next";
import { locales } from "@/i18n";

const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

const DISALLOWED_SEGMENTS = [
  "admin",
  "api",
  "login",
  "signup",
  "upload",
  "otp_verification",
  "change-password",
  "forgot-password",
];

function disallowPatterns(): string[] {
  const patterns: string[] = [];

  for (const segment of DISALLOWED_SEGMENTS) {
    // "en" is unprefixed under localePrefix: "as-needed", so the bare path
    // already covers it — only non-"en" locales need an explicit prefix.
    patterns.push(`/${segment}`);
    for (const locale of locales) {
      if (locale === "en") continue;
      patterns.push(`/${locale}/${segment}`);
    }
  }

  return patterns;
}

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: disallowPatterns(),
    },
    sitemap: new URL("/sitemap.xml", APP_URL).toString(),
  };
}
