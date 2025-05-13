"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const StatisticsPage = () => {
  const t = useTranslations("statistics");

  const [caseStudies, setStats] = useState([]);
  const [filteredStudies, setFilteredStudies] = useState([]);
  const [tagFilter, setTagFilter] = useState("");
  const [trimesterFilter, setTrimesterFilter] = useState("");
  const [pagefilter, setPageFilter] = useState("5");
  const [search, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [isDark, setIsDark] = useState(false); // 🌗 Theme toggle state

  useEffect(() => {
    searchUseCases({ searchTerm: "", searchMode: SEARCH_MODE.TITLE, category: CATEGORY.ALL }).then((res) => {
      const stats = getStats(res.filteredStudies);
      setStats(stats);
    });
  }, []);

  const searchUseCases = async (searchParams: SearchParams) => {
    const res = await fetch("/api/search-use-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchParams),
    });
    if (!res.ok) throw new Error(`Error: ${res.status}`);
    return res.json();
  };

  const getStats = (caseStudiesArray: CaseStudy[]) => {
    const tagCountMap = {};
    let total = 0;
    caseStudiesArray.forEach((cs) => {
      cs.tags.forEach((tag) => {
        const key = tag.trim().toLowerCase();
        tagCountMap[key] = (tagCountMap[key] || 0) + 1;
        total++;
      });
    });

    const stats = Object.keys(tagCountMap).map((tag, idx) => ({
      id: idx + 1,
      tag,
      publishNumber: tagCountMap[tag],
      popularity: `${((tagCountMap[tag] / total) * 100).toFixed(2)}%`,
      trimester: "2024 T1",
    }));

    setFilteredStudies(stats);
    return stats;
  };

  const tags = Array.from(new Set(caseStudies.map((s) => s.tag)));
  const trimesters = Array.from(new Set(caseStudies.map((s) => s.trimester)));

  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) filtered = filtered.filter((s) => s.tag === tagFilter);
    if (trimesterFilter) filtered = filtered.filter((s) => s.trimester === trimesterFilter);
    setFilteredStudies(filtered);
  }, [tagFilter, trimesterFilter, caseStudies]);

  const recordsPage = parseInt(pagefilter);
  const lastIndex = currentPage * recordsPage;
  const firstIndex = lastIndex - recordsPage;
  const records = filteredStudies.slice(firstIndex, lastIndex);
  const npage = Math.ceil(filteredStudies.length / recordsPage);

  const tri1 = caseStudies.filter((i) => i.trimester === "2024 T1").length;
  const tri2 = caseStudies.filter((i) => i.trimester === "2").length;
  const tri3 = caseStudies.filter((i) => i.trimester === "3").length;

  const chartData = {
    labels: [t("Trimester1"), t("Trimester2"), t("Trimester3")],
    datasets: [
      {
        label: t("Published Cases"),
        data: [tri1, tri2, tri3],
        backgroundColor: "#3EB470",
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className={`${isDark ? "dark" : ""}`}>
      <div className="font-sans bg-white dark:bg-gray-900 text-black dark:text-white min-h-screen flex flex-col transition-colors">
        <Header />

        <main className="max-w-7xl mx-auto px-4 py-8 flex-grow">
          {/* Theme Toggle */}
          <div className="flex justify-end mb-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <span className="text-xl">{isDark ? "🌙" : "☀️"}</span>
              <div className="relative inline-block w-12 align-middle select-none">
                <input
                  type="checkbox"
                  name="toggle"
                  id="toggle"
                  className="sr-only"
                  checked={isDark}
                  onChange={() => setIsDark(!isDark)}
                />
                <div className="block bg-gray-300 dark:bg-gray-700 w-12 h-6 rounded-full"></div>
                <div
                  className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition ${
                    isDark ? "translate-x-6" : ""
                  }`}
                ></div>
              </div>
            </label>
          </div>

          <h1 className="text-2xl font-bold mb-6 text-center">{t("Statistics")}</h1>

          <div className="flex flex-col md:flex-row gap-3 mb-4 justify-center">
            <select
              value={trimesterFilter}
              onChange={(e) => setTrimesterFilter(e.target.value)}
              className="text-sm border border-gray-300 rounded bg-gray-200 dark:bg-gray-700 dark:text-white px-3 py-2"
            >
              <option value="">{t("All Trimesters")}</option>
              {trimesters.map((trimester) => (
                <option key={trimester} value={trimester}>{trimester}</option>
              ))}
            </select>

            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              className="text-sm border border-gray-300 rounded bg-gray-200 dark:bg-gray-700 dark:text-white px-3 py-2"
            >
              <option value="">{t("All Tags")}</option>
              {tags.map((tag) => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>

            <div className="text-sm bg-gray-200 dark:bg-gray-700 border border-gray-300 rounded px-4 py-2">
              {t("Total Results")}: <strong>{filteredStudies.length}</strong>
            </div>
          </div>

          <input
            type="text"
            placeholder={t("Enter Tag name")}
            className="w-full text-sm border border-gray-300 rounded bg-gray-200 dark:bg-gray-700 dark:text-white px-3 py-2 mb-4"
            onChange={(e) => setSearchTerm(e.target.value)}
          />

          <div className="flex flex-col md:flex-row md:space-x-4">
            <div className="flex-1">
              <table className="w-full text-sm border border-gray-300 dark:border-gray-600">
                <thead className="bg-[#3EB470] text-white">
                  <tr>
                    <th className="py-2 px-3 text-left">{t("No")}</th>
                    <th className="py-2 px-3 text-left">{t("Tag")}</th>
                    <th className="py-2 px-3 text-left">{t("number")}</th>
                    <th className="py-2 px-3 text-left">{t("Popularity")}</th>
                  </tr>
                </thead>
                <tbody>
                  {records
                    .filter((item) =>
                      search === "" ? item : item.tag.toLowerCase().includes(search.toLowerCase())
                    )
                    .map((item, idx) => (
                      <tr key={item.id} className={idx % 2 === 0 ? "bg-white dark:bg-gray-800" : "bg-gray-100 dark:bg-gray-700"}>
                        <td className="py-2 px-3">{item.id}</td>
                        <td className="py-2 px-3 capitalize">{item.tag}</td>
                        <td className="py-2 px-3">{item.publishNumber}</td>
                        <td className="py-2 px-3">{item.popularity}</td>
                      </tr>
                    ))}
                </tbody>
              </table>

              <div className="flex justify-between items-center mt-3 text-sm">
                <span>
                  {firstIndex + 1}-{Math.min(lastIndex, filteredStudies.length)} of {filteredStudies.length}
                </span>
                <div className="flex items-center gap-3">
                  <button
                    disabled={currentPage <= 1}
                    onClick={() => setCurrentPage((prev) => prev - 1)}
                    className="px-3 py-1 bg-[#3EB470] text-white rounded disabled:opacity-50"
                  >
                    Prev
                  </button>
                  <span>Page {currentPage}</span>
                  <button
                    disabled={currentPage >= npage}
                    onClick={() => setCurrentPage((prev) => prev + 1)}
                    className="px-3 py-1 bg-[#3EB470] text-white rounded disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
                <select
                  value={pagefilter}
                  onChange={(e) => setPageFilter(e.target.value)}
                  className="border border-gray-300 rounded px-2 py-1"
                >
                  {[5, 10, 20, 30].map((val) => (
                    <option key={val} value={val}>{val}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="w-full md:w-1/3 mt-6 md:mt-0">
              <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-4">
                <h3 className="text-sm font-semibold mb-2">{t("Most Published Trimester")}</h3>
                <div className="h-64">
                  <Bar
                    data={chartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </main>

        <Footer />
      </div>
    </div>
  );
};

export default StatisticsPage;
