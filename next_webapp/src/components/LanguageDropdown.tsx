"use client";
import { usePathname, useRouter } from "@/i18n-navigation";
import React, { useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";

const LanguageDropdown: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const ref = useRef<HTMLDivElement>(null);

  const router = useRouter();
  const pathname = usePathname();

  const languages = [
    { locale: "en", label: "English" },
    { locale: "cn", label: "Chinese (中文)" },
    { locale: "es", label: "Spanish (Español)" },
    { locale: "el", label: "Greek (Ελληνικά)" },
    { locale: "ar", label: "Arabic (العربية)" },
    { locale: "it", label: "Italian (Italiano)" },
    { locale: "hi", label: "Hindi (हिन्दी)" },
    { locale: "vi", label: "Vietnamese (Tiếng Việt)" },
  ];

  // Custom hook to handle clicks outside the dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const selectLanguage = (locale: string) => {
    router.push(pathname, { locale });
    router.refresh();
  };

  const t = useTranslations("common");

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen ? "true" : "false"}
        aria-controls="language-dropdown"
        className="mr-2 bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-md text-sm font-medium dark:bg-gray-900 dark:hover:bg-green-600 dark:hover:text-gray-900 hover:bg-green-700 hover:text-white"
      >
        {t("Language")}
      </button>
      {isOpen && (
        <div
          id="language-dropdown"
          className="absolute z-10 bg-white rounded shadow-lg mt-1 w-48 dark:bg-gray-800"
        >
          {languages.map((language) => (
            <a
              key={language.locale}
              href="#"
              onClick={(event) => {
                event.preventDefault();
                selectLanguage(language.locale);
              }}
              className="block font-serif px-4 py-2 text-lg rounded text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600"
            >
              {language.label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
