"use client";

import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar, { LocalSearchMode } from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, CaseStudy } from "../../types";
import Tooglebutton from "../Tooglebutton/Tooglebutton";
import { demoCaseStudies } from "./database";

const UseCases: React.FC = () => {
  const [allCaseStudies] = useState<CaseStudy[]>(demoCaseStudies);
  const [filteredCaseStudies, setFilteredCaseStudies] =
    useState<CaseStudy[]>(demoCaseStudies);
  const [selectedCaseStudy, setSelectedCaseStudy] =
    useState<CaseStudy | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
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

  const handleToggle = (value: boolean) => {
    setDarkMode(value);
    localStorage.setItem("theme", value ? "dark" : "light");
  };

  const handleSearch = (
    term: string,
    mode: LocalSearchMode,
    cat: CATEGORY
  ) => {
    const keyword = term.trim().toLowerCase();

    if (!keyword) {
      setFilteredCaseStudies(allCaseStudies);
      setSelectedCaseStudy(null);
      return;
    }

    const filtered = allCaseStudies.filter((study) => {
      if (mode === "title") {
        return study.name?.toLowerCase().includes(keyword);
      }

      if (mode === "tag") {
        return study.tags?.some((tag) => tag.toLowerCase().includes(keyword));
      }

      if (mode === "content") {
        return study.description?.toLowerCase().includes(keyword);
      }

      return true;
    });

    setFilteredCaseStudies(filtered);
    setSelectedCaseStudy(null);
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#f7f9fb] text-black transition-all duration-300 dark:bg-gray-900 dark:text-white">
      <Header />

      <main className="flex-grow">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-10">
          <section className="mb-8 rounded-[28px] border border-gray-200 bg-white px-6 py-8 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:px-8">
            <div className="mb-4 inline-flex items-center rounded-full bg-green-50 px-4 py-1.5 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
              Open Data Use Cases
            </div>

            <div className="max-w-3xl">
              <h1 className="mb-3 text-4xl font-bold tracking-tight sm:text-5xl">
                Use Cases
              </h1>
            </div>
          </section>

          {!selectedCaseStudy && <SearchBar onSearch={handleSearch} />}

          <PreviewComponent
            caseStudies={filteredCaseStudies}
            trendingCaseStudies={filteredCaseStudies}
            selectedCaseStudy={selectedCaseStudy}
            onSelectCaseStudy={setSelectedCaseStudy}
            onBack={() => setSelectedCaseStudy(null)}
          />
        </div>
      </main>

      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleToggle} />
      </div>

      <Footer />
    </div>
  );
};

export default UseCases;