"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React, { useEffect, useState } from "react";
import { useTranslations } from "next-intl";

const Licensing = () => {
  const t = useTranslations("licensing");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    const isDark = savedTheme === "dark";
    setDarkMode(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const handleThemeChange = (isDark: boolean) => {
    setDarkMode(isDark);
    document.documentElement.classList.toggle("dark", isDark);
    localStorage.setItem("theme", isDark ? "dark" : "light");
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-100 dark:bg-black text-black dark:text-white transition-colors">
      <Header />

      <main className="flex-grow px-6 md:px-10 lg:px-20 py-10">
        <h1 className="text-4xl font-bold text-center mb-10">{t("Licensing")}</h1>

        {/* Section 1 */}
        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-2">{t("t1")}</h2>
          <p className="text-base mb-4 dark:text-gray-300">{t("p1")}</p>
          <ul className="list-disc pl-6 text-base dark:text-gray-300 space-y-1">
            <li>{t("p2")}</li>
            <li>{t("p3")}</li>
            <li>{t("p4")}</li>
            <li>{t("p5")}</li>
          </ul>
        </section>

        {/* Section 2 */}
        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-2">{t("t2")}</h2>
          <p className="text-base mb-4 dark:text-gray-300">{t("p6")}</p>
          <ul className="list-disc pl-6 text-base dark:text-gray-300 space-y-1">
            <li>{t("p7")}</li>
            <li>{t("p8")}</li>
            <li>{t("p9")}</li>
          </ul>
        </section>

        {/* Section 3 */}
        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-2">{t("t4")}</h2>
          <p className="text-base dark:text-gray-300">{t("p11")}</p>
        </section>

        {/* Section 4 */}
        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-2">{t("t5")}</h2>
          <p className="text-base mb-2 dark:text-gray-300">{t("p12")}</p>
          <ul className="list-disc pl-6 text-base dark:text-gray-300 space-y-1">
            <li>{t("p13")}</li>
            <li>{t("p14")}</li>
            <li>{t("p15")}</li>
          </ul>
        </section>

        {/* Section 5 */}
        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-2">{t("t3")}</h2>
          <p className="text-base dark:text-gray-300">{t("p10")}</p>
        </section>

        {/* Contact Info */}
        <div className="text-center mt-10">
          <p className="text-base dark:text-gray-300">
            {t("p16")}{" "}
            <a
              href="mailto:licensing@MOP.com.au"
              className="text-blue-600 dark:text-blue-400 underline"
            >
              licensing@MOP.com.au
            </a>
          </p>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Licensing;
