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
    <section className={`${bgClass} py-16 sm:py-20 lg:py-24 transition-colors duration-300`}>
      <div className="max-w-7xl mx-auto px-6 sm:px-8 md:px-10 lg:px-12">
        <div className="flex flex-col md:flex-row items-center gap-10 lg:gap-16 min-h-[420px]">
          {/* Image */}
          <div className="w-full md:w-1/2">
            <div className="group overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-500">
              <img
                src={imageSrc}
                alt={imageAlt}
                className="w-full aspect-[16/10] sm:aspect-video object-cover rounded-2xl transform transition-transform duration-500 group-hover:scale-105"
              />
            </div>
          </div>

          {/* Text Content */}
          <div className="w-full md:w-1/2 flex flex-col justify-center text-center md:text-left">
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight mb-4 text-black dark:text-white leading-tight">
              {title}
            </h2>
            <p className="text-[1rem] sm:text-[1.05rem] lg:text-[1.1rem] text-justify tracking-wide text-black/80 dark:text-white/90 leading-8 max-w-2xl mx-auto md:mx-0">
              {text}
            </p>
          </div>
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
