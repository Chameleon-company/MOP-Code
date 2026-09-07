import { locales } from "@/i18n";

const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

export interface HreflangAlternates {
  canonical: string;
  languages: Record<string, string>;
}

function normalizePathname(pathname: string): string {
  if (!pathname || pathname === "/") return "";
  return pathname.startsWith("/") ? pathname : `/${pathname}`;
}

/**
 * Builds an absolute URL for `locale` + `pathname`, respecting the app's
 * localePrefix: "as-needed" routing — "en" is unprefixed, all other locales
 * are prefixed with "/{locale}".
 */
export function getLocalizedUrl(locale: string, pathname: string): string {
  const path = normalizePathname(pathname);
  const localizedPath = locale === "en" ? path || "/" : `/${locale}${path}`;
  return new URL(localizedPath, APP_URL).toString();
}

/**
 * Given a pathname (without any locale prefix, e.g. "/about"), returns the
 * canonical URL (the unprefixed "en" URL) and an alternates.languages map
 * covering all supported locales, for use in Next.js `metadata.alternates`.
 */
export function getHreflangAlternates(pathname: string): HreflangAlternates {
  const languages: Record<string, string> = {};

  for (const locale of locales) {
    languages[locale] = getLocalizedUrl(locale, pathname);
  }

  // x-default points search engines at the unprefixed (English) URL.
  languages["x-default"] = getLocalizedUrl("en", pathname);

  return {
    canonical: getLocalizedUrl("en", pathname),
    languages,
  };
}
