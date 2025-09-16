"use client";
import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainImage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState } from "react";
import { ArrowLeft, FileText } from "lucide-react";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import HeroCarousel from "./HeroCarousel";


const style = `
.main-container {
  background: white;
  color: #263238;
  display: flex;
  flex-direction: column;
  margin-bottom: 0;
}
.dark .main-container {
  background: #263238;
  color: white;
  flex: 1;
}
.hero-section {
  width: 100%;
  overflow: hidden;
  position: relative;
}
.hero-section img {
  width: 100%;
  height: auto;
  max-height: 40rem;
  object-fit: cover;
  display: block;
}
.our-vision-section {
  display: flex;
  flex-direction: column;
  background: white;
  color: #263238;
  margin: 3rem auto;
  gap: 2rem;
  padding: 2rem;
  width: 90%;
  max-width: 1200px;
  box-sizing: border-box;
}
.dark .our-vision-section {
  background: #263238;
  color: white;
}
@media (min-width: 768px) {
  .our-vision-section {
    flex-direction: row;
    height: auto;
    min-height: 10rem;
  }
}
.img-div {
  width: 100%;
  height: 100%;
}
.img-div img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.text-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.our-vision {
  font-weight: 900;
  font-size: 2rem;
}
.case-studies-wrapper {
  background-color: #F6F9FC;
  padding: 2rem;
}
.dark .case-studies-wrapper {
  background-color: rgb(46, 38, 38);
}
.recent-case-studies {
  margin: 0 auto;
  padding: 0 2rem;
  max-width: 1200px;
  text-align: center;
  color: #263238;
}
.dark .recent-case-studies {
  color: white;
}
.recent-case-studies h2 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
}
.recent-case-studies p {
  font-size: 1rem;
  margin-bottom: 3rem;
  width: 50%;
  margin-left: auto;
  margin-right: auto;
}
.case-studies {
  margin: 2rem auto 0 auto;
  padding: 0 1rem;
  max-width: 1400px;
  color: #263238;
}
.dark .case-studies {
  color: white;
}
.case-study {
  background: white;
  color: #263238;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: box-shadow 0.3s;
  margin: 0 0.5rem;
  height: 80%;
  border-radius: 0.5rem;
}
.dark .case-study {
  background: #37474f;
  color: white;
}
.case-study:hover {
  box-shadow: 0 8px 12px rgba(0,0,0,0.2);
}
.case-study img {
  width: 100%;
  height: auto;
  object-fit: cover;
}
.react-multi-carousel-item {
  padding: 0 0.5rem;
}
`;

const categories = [
  { icon: "🏢", label: "EV Infrastructure" },
  { icon: "🅿️", label: "Parking" },
  { icon: "🚨", label: "Safety" },
];

const openPage = (label: string) => {
  if (label === "EV Infrastructure") {
    window.location.href = "/en/ev-infrastructure";
  }
};

const responsive = {
  superLargeDesktop: { breakpoint: { max: 4000, min: 3000 }, items: 5 },
  desktop: { breakpoint: { max: 3000, min: 1024 }, items: 3 },
  tablet: { breakpoint: { max: 1024, min: 464 }, items: 2 },
  mobile: { breakpoint: { max: 464, min: 0 }, items: 1 },
};

const Dashboard = () => {
  const t = useTranslations("common");
  const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>([]);
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(undefined);

  useEffect(() => {
    handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
  }, []);

  const searchUseCases = async (searchParams: SearchParams) => {
    const response = await fetch("/api/search-use-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchParams),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  };

  const handleSearch = async (searchTerm: string, searchMode: SEARCH_MODE, category: CATEGORY) => {
    const res = await searchUseCases({ searchTerm, searchMode, category });
    setFilteredCaseStudies(res?.filteredStudies);
  };

  const handleCaseStudyClick = (study: CaseStudy) => {
    setSelectedCaseStudy(study);
  };

  const handleBack = () => {
    setSelectedCaseStudy(undefined);
  };

  if (selectedCaseStudy) {
    return (
      <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900 text-black dark:text-white p-8">
        <button
          onClick={handleBack}
          className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
        >
          <ArrowLeft size={24} className="mr-2" />
          Back
        </button>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md flex-grow overflow-hidden flex flex-col">
          <h1 className="text-3xl font-bold mb-4 px-6 pt-6">{selectedCaseStudy.name}</h1>
          <iframe
            src={`/api?filename=${selectedCaseStudy.filename}`}
            title={selectedCaseStudy.name}
            className="flex-grow w-full border-none bg-white dark:bg-gray-900 text-black dark:text-white"
          />
        </div>
      </div>
    );
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: style }} />
      <div className="main-wrapper bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
        <div className="main-container">
          <section className="hero-section">
             {/*<Image src={mainimage} alt="main image1" />*/}
            <HeroCarousel />
          </section>

          <section className="our-vision-section">
            <div className="img-div">
              <Image src={secondimage} alt="Second Image" />
            </div>
            <div className="text-container">
              <h2 className="our-vision">{t("Our Vision")}</h2>
              <p>{t("intro")}</p>
            </div>
          </section>

          <section className="w-full max-w-6xl mx-auto mt-10 px-2 md:px-0">
            <h3 className="text-xl font-semibold mb-3">Explore by category</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-4 mb-2">
                  {categories.map((cat, idx) => (
                <div
                  key={idx}
                  onClick={() => openPage(cat.label)}
                  className="rounded-2xl border shadow-sm py-6 flex flex-col items-center bg-white dark:bg-gray-800 hover:shadow-md transition cursor-pointer"
                >
                  <span className="text-3xl mb-2">{cat.icon}</span>
                  <span className="text-sm font-medium text-center">{cat.label}</span>
                </div>
                ))}
              </div>
          </section>

          <section className="case-studies-wrapper">
            <section className="recent-case-studies">
              <h2>{t("Recent Case Studies")}</h2>
              <p>{t("p2")}</p>
            </section>

            <section className="case-studies">
              <Carousel responsive={responsive}>
                {filteredCaseStudies.slice(0, 6).map((study) => (
                  <div
                    key={study.id}
                    className="case-study p-4 cursor-pointer hover:shadow-2xl transition-shadow duration-300"
                    onClick={() => handleCaseStudyClick(study)}
                  >
                    <div className="flex items-center justify-center mb-4">
                      <FileText size={48} className="text-green-500" />
                      <FileText size={48} className="text-teal-400 -ml-6" />
                      <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
                    </div>
                    <h3 className="font-bold text-lg text-center mb-2">{study.name}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-sm text-center mb-2">
                      {study.description.split(" ").length > 50
                        ? `${study.description.split(" ").slice(0, 50).join(" ")}...`
                        : study.description}
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Tags:</p>
                      {study.tags.map((tag, index) => (
                        <span
                          key={index}
                          className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-xs px-2 py-1 rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </Carousel>
            </section>
          </section>
        </div>
      </div>
    </>
  );
};

export default Dashboard;
