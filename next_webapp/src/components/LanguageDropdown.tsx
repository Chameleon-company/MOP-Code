"use client";
import React, { useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";
import Language from "./language";

interface HeaderRightProps {
  language?: boolean;
  navOtherClass?: string;
}

// Define the LanguageDropdown component
const LanguageDropdown: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const ref = useRef<HTMLDivElement>(null);
  const t = useTranslations("common");

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

  // Function to switch language (simulating here, you would need to implement this depending on your app's context)
  const switchLanguage = (lang: string) => {
    console.log("Switching language to: ", lang);
    window.location.href = `/${lang}`;
    // Here you would put your language switching logic, possibly updating some global state or using a router to change the locale
  };

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="language-btn font-serif py-3 px-6 mx-3 text-white rounded-full text-lg"
      >
        {t("Language")}
      </button>
      {isOpen && (
        <div className="absolute z-10 bg-white rounded shadow-lg mt-1 w-48">
          <a
            href="#"
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
            onClick={() => switchLanguage("en")}
          >
            {t("en")}
          </a>
          <a
            href="#"
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
            onClick={() => switchLanguage("cn")}
          >
            {t("cn")}
          </a>
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
