// "use client";
// import React, { useState, useEffect } from "react";
// import Header from "../../../components/Header";
// import Footer from "../../../components/Footer";
// import SearchBar from "./searchbar";
// import PreviewComponent from "./preview";
// import { CATEGORY, SEARCH_MODE, SearchParams, CaseStudy } from "../../types";
// import { useTranslations } from "next-intl";

// async function searchUseCases(searchParams: SearchParams) {
//   const response = await fetch("/api/search-use-cases", {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//     },
//     body: JSON.stringify(searchParams),
//   });

//   if (!response.ok) {
//     throw new Error(`HTTP error! status: ${response.status}`);
//   }

//   return await response.json();
// }

// const UseCases = () => {
//   const [caseStudies, setCaseStudies] = useState([])
//   const [filteredCaseStudies, setFilteredCaseStudies] = useState(caseStudies);

//   useEffect(() => {
//     handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL)
//   }, [])

//   const handleSearch = async (
//     searchTerm: string,
//     searchMode: SEARCH_MODE,
//     category: CATEGORY
//   ) => {
//     const res = await searchUseCases({ searchTerm, searchMode, category });
//     console.log("🚀 ~ UseCases ~ res:", res);
//     setFilteredCaseStudies(res?.filteredStudies);
//   };

//   const t = useTranslations("usecases");

//   return (
//     <div className="font-sans bg-gray-100">
//       <Header />
//       <main>
//         <div className="app">
//           <section className="px-10 pt-5">
//             <p>
//               <span className="text-4xl font-bold text-black">
//                 {t("User Cases")}
//               </span>
//             </p>
//             <SearchBar onSearch={handleSearch} />
//             <PreviewComponent caseStudies={filteredCaseStudies} />
//           </section>
//         </div>
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default UseCases;

//Divyanga C.S.Lokuhetti #s223590519
//Team Project(B) - T1 2025
"use client";
import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, SEARCH_MODE, SearchParams, CaseStudy } from "../../types";
import { useTranslations } from "next-intl";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

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

  const handleSearch = async (
    searchTerm: string,
    searchMode: SEARCH_MODE,
    category: CATEGORY
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
