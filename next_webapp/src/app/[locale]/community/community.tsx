"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/community.css";
import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";

const Community = () => {
  const t = useTranslations("community");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  }, [darkMode]);

  const FacilityCard = ({
    imageSrc,
    title,
  }: {
    imageSrc: string;
    title: string;
  }) => (
    <div className="facility-card">
      <img src={imageSrc} alt={title} className="facility-img" />
      <p className="facility-title">{title}</p>
    </div>
  );

  return (
    <div className="bg-white dark:bg-[#1d1919] text-black dark:text-white min-h-screen transition-colors duration-300">
      <Header />

      {/* Hero Image */}
      <div className="hero-section">
        <img
          src="/img/melbourne-skyline.jpg"
          alt="Melbourne Skyline"
          className="hero-img"
        />
        <h1 className="hero-title">{t("Community and Social Facilities")}</h1>
      </div>

      {/* Categories Button */}
      <div className="categories-btn-wrapper">
        <button className="categories-btn">{t("Categories")}</button>
      </div>

      {/* Civic & Administrative Facilities */}
      <section className="facilities-section">
        <h2 className="section-title">{t("Civic and Administrative Facilities")}</h2>
        <p className="section-subtitle">
          {t("See Melbourne’s historic Town Hall, Parliament House, and the grand State Library.")}
        </p>

        <div className="facilities-grid">
          <FacilityCard imageSrc="/img/townhall.jpg" title={t("Melbourne Town Hall")} />
          <FacilityCard imageSrc="/img/treasury.png" title={t("Melbourne Treasury")} />
          <FacilityCard imageSrc="/img/fedsq.jpg" title={t("Federation Square")} />
          <FacilityCard imageSrc="/img/parliament.png" title={t("Parliament of Victoria")} />
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Community;
