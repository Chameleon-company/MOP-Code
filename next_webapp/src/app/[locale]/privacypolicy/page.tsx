"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { HiMoon, HiSun } from "react-icons/hi2";
import "../../../../public/styles/privacy.css";

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");

  const [openSections, setOpenSections] = useState<{ [key: string]: boolean }>({});
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    if (stored === "dark") setIsDarkMode(true);
  }, []);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const toggleTheme = () => setIsDarkMode((prev) => !prev);

  const sections = [
    { key: "1", title: t("t1"), content: t("p1") },
    { key: "2", title: t("t2"), content: t("p2") },
    { key: "3", title: t("t3"), content: t("p3") },
    { key: "4", title: t("t4"), content: t("p4") },
    { key: "5", title: t("t5"), content: t("p5") },
    { key: "6", title: t("t6"), content: t("p6") },
  ];

  return (
    <div className="flex flex-col min-h-screen bg-white text-gray-900 dark:bg-black dark:text-white transition-colors duration-300">
      <Header />

      <main className="flex-grow flex flex-col items-center font-montserrat relative pb-20">
        <h1 className="text-3xl font-bold mt-10 mb-6">{t("Privacy Policy")}</h1>

        <div className="w-full max-w-3xl px-4 rounded-lg p-6 bg-gray-200 text-gray-900 dark:bg-[#263238] dark:text-white">
          {sections.map(({ key, title, content }) => (
            <div key={key} className="mb-2">
              <button
                onClick={() => toggleSection(key)}
                className="w-full flex justify-between items-center font-bold px-4 py-3 rounded-sm transition bg-[#2ECC71] text-black hover:bg-[#2abb67] dark:bg-[#2ECC71] dark:hover:bg-[#2abb67]"
              >
                <span>{title}</span>
                <span>{openSections[key] ? "▲" : "▼"}</span>
              </button>
              {openSections[key] && (
                <div className="p-4 text-sm rounded-b-sm bg-green-200 text-black dark:bg-[#acecc7]">
                  {content}
                </div>
              )}
            </div>
          ))}
        </div>

        <p className="text-center text-xs text-gray-700 mt-10 w-[80%] dark:text-gray-400">
          {t("p7")}
        </p>

        <button
          onClick={toggleTheme}
          className="absolute bottom-5 right-5 p-3 bg-[#f0f0f0] rounded-full shadow-md hover:bg-[#e0e0e0] dark:bg-[#333333] dark:hover:bg-[#444444] transition"
          aria-label="Toggle Theme"
        >
          {isDarkMode ? (
            <HiSun className="text-yellow-400 text-xl" />
          ) : (
            <HiMoon className="text-gray-800 text-xl" />
          )}
        </button>
      </main>

      <Footer />
    </div>
  );
};

export default Privacypolicy;
