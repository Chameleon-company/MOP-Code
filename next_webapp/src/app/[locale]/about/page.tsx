"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";

const About = () => {
  const t = useTranslations("about");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  }, [darkMode]);

  const Section = ({
    imageSrc,
    imageAlt,
    title,
    text,
    bgClass,
  }: {
    imageSrc: string;
    imageAlt: string;
    title: string;
    text: string;
    bgClass: string;
  }) => (
    <section className={`${bgClass} py-12`}>
      <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row items-center md:items-center gap-8 min-h-[400px]">
        {/* Image */}
        <div className="w-full md:w-1/2">
          <img
            src={imageSrc}
            alt={imageAlt}
            className="w-full aspect-video object-cover rounded-lg shadow-md"
          />
        </div>

        {/* Text Content */}
        <div className="w-full md:w-1/2 flex flex-col justify-center">
          <h2 className="text-2xl font-bold mb-4 text-black dark:text-white">{title}</h2>
          <p className="text-base text-justify text-black dark:text-white leading-relaxed">
            {text}
          </p>
        </div>
      </div>
    </section>
  );

  return (
    <div className="bg-white dark:bg-[#1d1919] text-black dark:text-white min-h-screen transition-colors duration-300">
      <Header />

      {/* Section 1: About Us – light: white | dark: #263238 */}
      <Section
        imageSrc="/img/mel.jpg"
        imageAlt="Melbourne Open Playground"
        title={t("About Us")}
        text={t("p2")}
        bgClass="bg-white dark:bg-[#263238]"
      />

      {/* Section 2: Open Data Leadership – light: green-500 | dark: #14532d */}
      <Section
        imageSrc="/img/leadership.png"
        imageAlt="Leadership Image"
        title={t("Open Data Leadership")}
        text={t("p3")}
        bgClass="bg-green-500 dark:bg-[#14532d]"
      />

      {/* Section 3: Our Goals – light: white | dark: #263238 */}
      <Section
        imageSrc="/img/goals.png"
        imageAlt="Our Goals"
        title={t("Our Goals")}
        text={t("p4")}
        bgClass="bg-white dark:bg-[#263238]"
      />

      <Footer />
    </div>
  );
};

export default About;