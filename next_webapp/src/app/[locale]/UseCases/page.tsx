// app/[locale]/UseCases/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, SEARCH_MODE, SearchParams, CaseStudy } from "../../types";

async function searchUseCases(params: SearchParams) {
  const res = await fetch("/api/search-use-cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

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
      <Footer />
    </div>
  );
};

export default UseCases;
