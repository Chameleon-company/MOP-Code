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
    router.push(pathname, { locale });
    router.refresh();
  };

  const t = useTranslations("common");

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="mr-2 bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-md text-sm font-medium dark:bg-gray-900 dark:hover:bg-green-600 dark:hover:text-gray-900 hover:bg-green-700 hover:text-white"
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
            中文
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("es");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Español
          </a>
          <a
            href="#"
            onClick={(event) => {
              event.preventDefault();
              selectLanguage("el");
            }}
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Ελληνικά
          </a>
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
