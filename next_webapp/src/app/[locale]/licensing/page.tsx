"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React, { useEffect } from "react";
import { useTranslations } from "next-intl";
import Tooglebutton from "./Tooglebutton";

const Licensing = () => {
  const t = useTranslations("licensing");

  const handleThemeChange = (isDark: boolean) => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
      document.documentElement.classList.add("dark");
    }
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-gray-100 dark:bg-black">
      <Header />
      <main className="flex-grow px-6 md:px-10 lg:px-20 py-10">
        <h1 className="text-black dark:text-white text-4xl font-bold text-center mb-8">
          {t("Licensing")}
        </h1>

        <section>
          <h2 className="text-black dark:text-white text-xl font-semibold mb-2">{t("t1")}</h2>
          <p className="text-black dark:text-gray-300 text-base mb-4">{t("p1")}</p>
          <ul className="list-disc pl-6 text-black dark:text-gray-300 text-base mb-8">
            <li>{t("p2")}</li>
            <li>{t("p3")}</li>
            <li>{t("p4")}</li>
            <li>{t("p5")}</li>
          </ul>
        </section>

        <section>
          <h2 className="text-black dark:text-white text-xl font-semibold mb-2">{t("t2")}</h2>
          <p className="text-black dark:text-gray-300 text-base mb-4">{t("p6")}</p>
          <ul className="list-disc pl-6 text-black dark:text-gray-300 text-base mb-8">
            <li>{t("p7")}</li>
            <li>{t("p8")}</li>
            <li>{t("p9")}</li>
          </ul>
        </section>

        <section>
          <h2 className="text-black dark:text-white text-xl font-semibold mb-2">{t("t4")}</h2>
          <p className="text-black dark:text-gray-300 text-base mb-6">{t("p11")}</p>
        </section>

        <section>
          <h2 className="text-black dark:text-white text-xl font-semibold mb-2">{t("t5")}</h2>
          <p className="text-black dark:text-gray-300 text-base mb-2">{t("p12")}</p>
          <ul className="list-disc pl-6 text-black dark:text-gray-300 text-base mb-8">
            <li>{t("p13")}</li>
            <li>{t("p14")}</li>
            <li>{t("p15")}</li>
          </ul>
        </section>

        <section>
          <h2 className="text-black dark:text-white text-xl font-semibold mb-2">{t("t3")}</h2>
          <p className="text-black dark:text-gray-300 text-base mb-8">{t("p10")}</p>
        </section>

        <div className="text-center mt-10">
          <p className="text-black dark:text-gray-300 text-base">
            {t("p16")}{" "}
            <a href="mailto:licensing@MOP.com.au" className="text-blue-600 dark:text-blue-400">
              licensing@MOP.com.au
            </a>
          </p>
        </div>
      </main>

      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleThemeChange} />
      </div>

      <Footer />
    </div>
  );
};

export default Licensing;
