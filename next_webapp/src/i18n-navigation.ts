import { createSharedPathnamesNavigation } from "next-intl/navigation";
import { locales } from "./i18n";

export const { usePathname, useRouter, Link, redirect } =
  createSharedPathnamesNavigation({
    localePrefix: "as-needed",
    locales,
  });
