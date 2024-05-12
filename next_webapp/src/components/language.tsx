"use client";

import { useLocale } from "next-intl";

import { AppConfig } from "../utils/AppConfig";
import { usePathname, useRouter } from "../utils/i18nNavigation";

export default function Language() {
  const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();

  const handleChange = (lang: string) => {
    router.push(pathname, { locale: lang });
    router.refresh();
  };

  return (
    <li className="nav-item dropdown language-select text-uppercase">
      <a
        aria-expanded="false"
        aria-haspopup="true"
        className="nav-link dropdown-item dropdown-toggle"
        data-bs-toggle="dropdown"
        role="button"
      >
        {locale}
      </a>

      <ul className="dropdown-menu">
        {AppConfig.locales.map((lang) => (
          <li className="nav-item" key={lang}>
            <button
              className="dropdown-item"
              onClick={() => handleChange(lang)}
            >
              {lang.toUpperCase()}
            </button>
          </li>
        ))}
      </ul>
    </li>
  );
}
