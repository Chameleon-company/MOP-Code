"use client";
import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar from "./searchbar";
import PreviewComponent from "./preview";
import { caseStudies } from "./database";
import { CATEGORY, SEARCH_MODE, SearchParams } from "../../types";
import { useTranslations } from "next-intl";
// import path from "path";
// import fs from "fs";

async function searchUseCases(searchParams: SearchParams) {
  const response = await fetch("/api/search-use-cases", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(searchParams),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

const UseCases = () => {
  const [filteredCaseStudies, setFilteredCaseStudies] = useState(caseStudies);

  const handleSearch = async (
    searchTerm: string,
    searchMode: SEARCH_MODE,
    category: CATEGORY
  ) => {
    const res = await searchUseCases({ searchTerm, searchMode, category });
    console.log("🚀 ~ UseCases ~ res:", res);
    setFilteredCaseStudies(res?.filteredStudies);
  };

  const t = useTranslations("usecases");

  return (
    <div className="font-sans bg-gray-100">
      <Header />
      <main>
        <div className="app">
          <section className="px-10 pt-5">
            <p>
              <span className="text-4xl font-bold text-black">
                {t("User Cases")}
              </span>
            </p>
            <SearchBar onSearch={handleSearch} />
            <PreviewComponent caseStudies={filteredCaseStudies} />
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default UseCases;
