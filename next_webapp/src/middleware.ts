import createMiddleware from "next-intl/middleware";
import { locales } from "./i18n";

export default createMiddleware({
  // A list of all locales that are supported
  locales,
  defaultLocale: "en",
  localePrefix: "as-needed", 
  localeDetection: true,
});

export const config = {
  // Match only internationalized pathnames
  matcher: ["/", `/(cn|en|es|el)/:path*`],
};
