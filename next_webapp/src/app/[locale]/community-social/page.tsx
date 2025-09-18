"use client";

import Link from "next/link";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";

const CommunitySocial = () => {
  const t = useTranslations("communitySocial");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  }, [darkMode]);

  const FacilityCard = ({
    imageSrc,
    title,
    link,
  }: {
    imageSrc: string;
    title: string;
    link: string;
  }) => (
    <Link href={link} target="_blank" rel="noopener noreferrer" className="w-full">
      <div className="facility-card flex flex-col items-center p-4 bg-white dark:bg-[#2a2a2a] rounded-lg shadow-md hover:shadow-lg transition cursor-pointer">
        <img
          src={imageSrc}
          alt={title}
          className="facility-img w-40 h-32 object-cover rounded-lg mb-3"
        />
        <p className="facility-title text-sm font-medium text-center">{title}</p>
      </div>
    </Link>
  );

  return (
    <div className="bg-white dark:bg-[#1d1919] text-black dark:text-white min-h-screen transition-colors duration-300">
      <Header />

      <div className="text-center py-6 bg-white dark:bg-[#1d1919]">
        <h1 className="text-3xl font-bold text-black dark:text-white">
          {t("title", { default: "Community and Social Facilities" })}
        </h1>
      </div>

      {/* Hero Image */}
      <div className="hero-section relative">
        <img
          src="/img/melb.jpg"
          alt="Melbourne"
          className="hero-img w-full h-[65vh]"
        />
      </div>

      {/* Categories Button */}
      <div className="categories-btn-wrapper flex justify-center my-6">
        <button className="categories-btn px-6 py-2 bg-gray-200 dark:bg-gray-700 rounded-full shadow hover:bg-gray-300 dark:hover:bg-gray-600 transition">
          {t("categories", { default: "Categories" })}
        </button>
      </div>

      {/* Civic & Administrative Facilities */}
      <section className="facilities-section px-6 py-4 bg-green-200 dark:bg-green-800">
        <h2 className="section-title text-lg font-bold mb-2">
          {t("civicTitle", { default: "Civic and Administrative Facilities" })}
        </h2>
        <p className="section-subtitle mb-4">
          {t("civicDesc", {
            default:
              "See Melbourne’s historic Town Hall, Parliament House, and the grand State Library.",
          })}
        </p>

      <div className="facilities-grid grid grid-cols-2 sm:grid-cols-4 gap-20">
          <FacilityCard
            imageSrc="/img/townhall.jpg"
            title={t("townHall", { default: "Melbourne Town Hall" })}
            link="https://whatson.melbourne.vic.gov.au/things-to-do/melbourne-town-hall"
          />
          <FacilityCard
            imageSrc="/img/treasury.jpg"
            title={t("treasury", { default: "Melbourne Treasury" })}
            link="https://www.oldtreasurybuilding.org.au/?srsltid=AfmBOorjA9iWzyPM6raLZPLigQGmiy0X2Qos-Bse8Li0KKCTMWmmudtZ"
          />
          <FacilityCard
            imageSrc="/img/fedsq.jpg"
            title={t("fedSquare", { default: "Federation Square" })}
            link="https://www.visitmelbourne.com/regions/melbourne/see-and-do/art-and-culture/architecture-and-design/federation-square"
          />
          <FacilityCard
            imageSrc="/img/parliament.jpg"
            title={t("parliament", { default: "Parliament of Victoria" })}
            link="https://www.parliament.vic.gov.au/"
          />
        </div>
      </section>

      {/* Educational Facilities */}
      <section className="educational-section px-6 py-4 bg-blue-200 dark:bg-blue-800">
        <h2 className="section-title text-lg font-bold mb-2">
          {t("educationalTitle", { default: "Educational Facilities" })}
        </h2>
        <p className="section-subtitle mb-4">
          {t("educationalDesc", {
            default: "Explore the Deakin University's Burwood campus and RMIT's striking architecture.",
          })}
        </p>

        <div className="facilities-grid grid grid-cols-2 sm:grid-cols-4 gap-20">
          <FacilityCard
            imageSrc="/img/library.jpg"
            title={t("vicLibrary", { default: "State Library of Victoria" })}
            link="https://www.slv.vic.gov.au/"
          />
          <FacilityCard
            imageSrc="/img/rmit.jpg"
            title={t("rmit", { default: "RMIT University" })}
            link="https://www.rmit.edu.au/"
          />
          <FacilityCard
            imageSrc="/img/deakin.jpg"
            title={t("deakin", { default: "Deakin University" })}
            link="https://www.deakin.edu.au/"
          />
          <FacilityCard
            imageSrc="/img/monash.jpg"
            title={t("monash", { default: "Monash University" })}
            link="https://www.monash.edu/"
          />
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default CommunitySocial;
