"use client";
import React, { useState, useEffect } from "react";
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import SearchBar from "./searchbar";
import PreviewComponent from "./preview";
import { caseStudies, CaseStudy, CATEGORY } from "./database";

const UseCases = () => {
  const [filteredCaseStudies, setFilteredCaseStudies] = useState(caseStudies);

  const handleSearch = (searchTerm: string, category: string) => {
    setFilteredCaseStudies(
      caseStudies.filter((caseStudy) => {
        return (
          (category === CATEGORY.ALL || category === caseStudy.category) &&
          caseStudy.title.toLowerCase().includes(searchTerm.toLowerCase())
        );
      })
    );
  };
  return (
    <div className="font-sans bg-gray-100">
      <Header />
      <main>
        <div className="app">
          <section>
            <p>
              <span className="text-4xl text-black">Case</span>
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
