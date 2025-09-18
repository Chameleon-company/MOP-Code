"use client";

import usecases from "../../utils/usecases.json";
import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import Tooglebutton from "../Tooglebutton/Tooglebutton";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { useLocale } from "next-intl";

// Types
interface UseCase {
  name: string;
  address: string;
}
interface Category {
  category: string;
  usecases: UseCase[];
}

// Category card component
const CategoryCard: React.FC<{ category: Category }> = ({ category }) => {
  const locale = useLocale();
  const slug = category.category.toLowerCase().replace(/\s+/g, "-");

  return (
    <Link href={`/${locale}/usecases/${slug}`} className="h-full">
      <div className="flex flex-col h-full max-h-80 bg-white dark:bg-gray-900 rounded-2xl shadow hover:shadow-lg transition cursor-pointer overflow-hidden">
        
        {/* Image placeholder (2/3 height) */}
        <div className="flex-grow basis-2/3 bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
          <span className="text-gray-500 text-sm">Image</span>
        </div>

        {/* Text content (1/3 height) */}
        <div className="flex-grow basis-1/3 p-4 flex flex-col items-center justify-center">
          <h2 className="text-lg text-green-600 font-semibold text-center mb-1">
            {category.category}
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-300 text-center">
            {category.usecases.length} use cases
          </p>
        </div>
      </div>
    </Link>
  );
};

const UseCases: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);
  const t = useTranslations("usecases");

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

  return (
    <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-gray-800 text-black dark:text-white transition-all duration-300">
      <Header />
      <main className="flex-grow">
        <div className="max-w-7xl mx-auto px-6 py-10 h-full">
          <h1 className="text-4xl font-bold mb-8">{t("User Cases")}</h1>

          {/* Category Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 h-[calc(100vh-16rem)]">
            {usecases.slice(0, 9).map((cat) => (
              <CategoryCard key={cat.category} category={cat} />
            ))}
          </div>
        </div>
      </main>

      {/* Dark Mode Toggle */}
      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleToggle} />
      </div>

      <Footer />
    </div>
  );
};


export default UseCases;
