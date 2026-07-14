"use client";
import { usePathname, useRouter } from "@/i18n-navigation";
import React, { useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";

const LanguageDropdown: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const ref = useRef<HTMLDivElement>(null);

  const router = useRouter();
  const pathname = usePathname();

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
    setIsOpen(false);
    router.push(pathname, { locale });
    router.refresh();
  };

  const t = useTranslations("common");

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
  aria-expanded={isOpen}
  className="mr-2 inline-flex h-10 min-w-[96px] items-center justify-center rounded-lg border border-green-600 bg-white px-4 text-sm font-medium text-green-600 transition-all duration-200 transform hover:scale-105 hover:bg-green-50 hover:text-green-700 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 dark:border-green-400 dark:bg-black dark:text-green-300 dark:hover:bg-gray-800"
>
        {t("Language")}
      </button>
      {isOpen && (
        <div className="absolute z-10 bg-white rounded shadow-lg mt-1 w-48">
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("en");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            English
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("cn");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Chinese (中文)
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("es");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Spanish (Español)
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("el");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Greek (Ελληνικά)
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("ar");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Arabic (العربية)
        </a>
        <a
          href="#"
          onClick={(event) => {
            event.preventDefault();
            selectLanguage("it");
          }}
          className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
        >
          Italian (Italiano)
        </a>
        <a
          href="#"
          onClick={(event) => {
            event.preventDefault();
            selectLanguage("hi");
          }}
          className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
        >
          Hindi (हिन्दी)
        </a>
        <a
          href="#"
          onClick={(event) => {
            event.preventDefault();
            selectLanguage("vi");
          }}
          className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
        >
          Vietnamese (Tiếng Việt)
        </a>

        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;