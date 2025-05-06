
// app/[locale]/UseCases/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, SEARCH_MODE, SearchParams, CaseStudy } from "../../types";
import { useTranslations } from "next-intl";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

async function searchUseCases(params: SearchParams) {
  const res = await fetch("/api/search-use-cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

const UseCases = () => {
  const [caseStudies, setCaseStudies] = useState<CaseStudy[]>([]);
  const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>([]);
  const [darkMode, setDarkMode] = useState(false);

  const t = useTranslations("usecases");

  useEffect(() => {
    handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);

    // Load dark mode preference
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme === "dark") {
      setDarkMode(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);
const UseCases: React.FC = () => {
  const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>(
    []
  );
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | null>(
    null
  );

  useEffect(() => {
    handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
  }, []);

  const handleSearch = async (
    term: string,
    mode: SEARCH_MODE,
    cat: CATEGORY
  ) => {
    const res = await searchUseCases({ searchTerm, searchMode, category });
    setFilteredCaseStudies(res?.filteredStudies || []);
  };

  const handleToggle = (value: boolean) => {
    setDarkMode(value);
    localStorage.setItem("theme", value ? "dark" : "light");
  };

  return (
    <div className="font-sans bg-gray-100 dark:bg-[#1d1919] min-h-screen text-black dark:text-white transition-all duration-300">
      <Header />

      <main>
        <section className="px-10 pt-5">
          <p>
            <span className="text-4xl font-bold">{t("User Cases")}</span>
          </p>
          <SearchBar onSearch={handleSearch} />
          <PreviewComponent caseStudies={filteredCaseStudies} />
        </section>
    try {
      const res = await searchUseCases({
        searchTerm: term,
        searchMode: mode,
        category: cat,
      });
      setFilteredCaseStudies(res.filteredStudies);
    } catch {
      setFilteredCaseStudies([]);
    }
  };

  return (
    <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-gray-800">
      <Header />
      <main className="flex-grow">
        <div className="max-w-7xl mx-auto px-4 lg:px-10 bg-white dark:bg-gray-800">
          <section className="py-5 bg-white dark:bg-gray-800">
            <h1 className="text-4xl font-bold text-black dark:text-white mb-6">
              Use Cases
            </h1>
            {!selectedCaseStudy && <SearchBar onSearch={handleSearch} />}
            <PreviewComponent
              caseStudies={filteredCaseStudies}
              trendingCaseStudies={filteredCaseStudies}
              selectedCaseStudy={selectedCaseStudy}
              onSelectCaseStudy={setSelectedCaseStudy}
              onBack={() => setSelectedCaseStudy(null)}
            />
          </section>
        </div>
      </main>

      {/* Toggle Dark Mode */}
      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleToggle} />
      </div>

      <Footer />
    </div>
  );
};

export default UseCases;
