"use client";
import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainimage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState, useRef } from "react";
import {
  ArrowLeft,
  FileText,
  ArrowRight,
  Play,
  ChevronDown,
  Search,
  X,
  Users,
  Car,
  Trees,
  Home,
  DollarSign,
  Heart,
} from "lucide-react";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import CityMetricCard, { CityMetric } from "@/components/CityMetricCard";
import HeroCarousel from "./HeroCarousel";

const cityMetrics: CityMetric[] = [
  {
    id: "1",
    title: "Population",
    value: "2.3M",
    change: 2.5,
    icon: <Users size={20} className="text-blue-700 dark:text-blue-300" />,
    category: "population",
  },
  {
    id: "2",
    title: "Public Transport",
    value: "78%",
    change: 5.2,
    icon: <Car size={20} className="text-purple-700 dark:text-purple-300" />,
    category: "transportation",
  },
  {
    id: "3",
    title: "Green Spaces",
    value: "32%",
    change: -1.2,
    icon: <Trees size={20} className="text-green-700 dark:text-green-300" />,
    category: "environment",
  },
  {
    id: "4",
    title: "Housing Affordability",
    value: "64%",
    change: 3.1,
    icon: <Home size={20} className="text-amber-700 dark:text-amber-300" />,
    category: "housing",
  },
  {
    id: "5",
    title: "Median Income",
    value: "$65,420",
    change: 4.7,
    icon: <DollarSign size={20} className="text-indigo-700 dark:text-indigo-300" />,
    category: "economy",
  },
  {
    id: "6",
    title: "Life Expectancy",
    value: "81.2 yrs",
    change: 0.8,
    icon: <Heart size={20} className="text-pink-700 dark:text-pink-300" />,
    category: "health",
  },
];

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

// Get available search modes from the SEARCH_MODE enum
const getSearchModeValues = () => {
  try {
    const searchModes = Object.values(SEARCH_MODE).filter(
      (value) => typeof value === "string"
    ) as string[];
    return searchModes.length > 0 ? searchModes : ["TITLE", "CONTENT"];
  } catch (error) {
    return ["TITLE", "CONTENT"];
  }
};

const Dashboard = () => {
  const t = useTranslations("common");
  const t_hero = useTranslations("hero");

  const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>([]);
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(undefined);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchMode, setSearchMode] = useState<SEARCH_MODE>(SEARCH_MODE.TITLE);
  const [category, setCategory] = useState<CATEGORY>(CATEGORY.ALL);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [debugInfo, setDebugInfo] = useState<any>(null);

  const searchContainerRef = useRef<HTMLDivElement>(null);
  const availableSearchModes = getSearchModeValues();

  useEffect(() => {
    handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
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

  const handleSearch = async (term: string, mode: SEARCH_MODE, cat: CATEGORY) => {
    setIsSearching(true);
    try {
      const res = await searchUseCases({ searchTerm: term, searchMode: mode, category: cat });
      setFilteredCaseStudies(res?.filteredStudies || []);
      setDebugInfo(res?.debug || null);
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleCaseStudyClick = (study: CaseStudy) => {
    setSelectedCaseStudy(study);
    setShowSearchResults(false);
  };

  const handleBack = () => setSelectedCaseStudy(undefined);

  const scrollToContent = () => {
    document.querySelector(".our-vision-section")?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(searchTerm, searchMode, category);
    setShowSearchResults(true);
  };

  const clearSearch = () => {
    setSearchTerm("");
    setShowSearchResults(false);
    handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
  };

  // Selected case study view
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

  // Main dashboard view
  return (
    <>
      <div className="main-wrapper bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
        <div className="main-container">
          <section className="hero-section">
            <HeroCarousel />
            <div className="scroll-indicator" onClick={scrollToContent}>
              <ChevronDown size={40} />
            </div>
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
                      {study.tags?.map((tag, index) => (
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
