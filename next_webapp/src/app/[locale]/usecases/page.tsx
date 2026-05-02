"use client";

import React, { useState, useEffect, useCallback } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar, { LocalSearchMode } from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY } from "../../types";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const PAGE_SIZE = 9;

const UseCases: React.FC = () => {
  const [usecases, setUsecases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchMode, setSearchMode] = useState<LocalSearchMode>("title");
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

  const fetchUseCases = useCallback(async (
    term: string,
    mode: LocalSearchMode,
    currentPage: number
  ) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
      });

      if (term) {
        if (mode === "title")   { params.set("search", term); params.set("search_by", "title"); }
        if (mode === "content") { params.set("search", term); params.set("search_by", "description"); }
        if (mode === "tag")     { params.set("tag_name", term); }
      }

      const res = await fetch(`/api/usecases?${params}`);
      const json = await res.json();
      if (json.success) {
        setUsecases(json.data || []);
        setTotal(json.pagination?.total ?? 0);
        setTotalPages(json.pagination?.totalPages ?? 1);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUseCases(searchTerm, searchMode, page);
  }, [searchTerm, searchMode, page, fetchUseCases]);

  const handleToggle = (value: boolean) => {
    setDarkMode(value);
    localStorage.setItem("theme", value ? "dark" : "light");
  };

  const handleSearch = (term: string, mode: LocalSearchMode, _cat: CATEGORY) => {
    setSearchTerm(term);
    setSearchMode(mode);
    setPage(1);
  };

  const mapped = usecases.map((u) => ({
    id: String(u.id),
    name: u.title,
    title: u.title,
    description: u.description ?? "",
    image: u.cover_img ?? "",
    tags: (u.tags ?? []).map((t: { name: string }) => t.name),
    category: String(u.category_id ?? ""),
    filename: "",
  }));

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

          <SearchBar onSearch={handleSearch} />

          {loading ? (
            <div className="flex justify-center py-20">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
            </div>
          ) : (
            <PreviewComponent
              caseStudies={mapped}
              page={page}
              totalPages={totalPages}
              total={total}
              onPageChange={setPage}
            />
          )}
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
