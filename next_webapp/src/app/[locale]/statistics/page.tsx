"use client";

import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Colors,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from "chart.js";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import Footer from "../../../components/Footer";
import Header from "../../../components/Header";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Statistics = () => {
  const [caseStudies, setStats] = useState([]);
  const [filteredStudies, setFilteredStudies] = useState([]);
  const [tagFilter, setTagFilter] = useState("");
  const [trimesterFilter, setTrimesterFilter] = useState("");
  const [pagefilter, setPageFilter] = useState("5");
  const [search, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const t = useTranslations("statistics");

  useEffect(() => {
    searchUseCases({ searchTerm: "", searchMode: SEARCH_MODE.TITLE, category: CATEGORY.ALL })
      .then((useCases) => {
        const stats = getStats(useCases.filteredStudies);
        setStats(stats);
        setFilteredStudies(stats);
      });
  }, []);

  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) filtered = filtered.filter((study) => study.tag === tagFilter);
    if (trimesterFilter) filtered = filtered.filter((study) => study.trimester === trimesterFilter);
    setFilteredStudies(filtered);
  }, [tagFilter, trimesterFilter, caseStudies]);

  async function searchUseCases(searchParams: SearchParams) {
    const response = await fetch("/api/search-use-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchParams),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  }

  const getStats = (caseStudiesArray: CaseStudy[]) => {
    const tagCountMap: { [key: string]: number } = {};
    let totalTags = 0;
    caseStudiesArray.forEach((caseStudy) => {
      caseStudy.tags.forEach((tag) => {
        const normalizedTag = tag.trim().toLowerCase();
        tagCountMap[normalizedTag] = (tagCountMap[normalizedTag] || 0) + 1;
        totalTags++;
      });
    });
    const tagPopularityArray = Object.keys(tagCountMap).map((tag, index) => ({
      id: index + 1,
      tag,
      publishNumber: tagCountMap[tag],
      popularity: `${((tagCountMap[tag] / totalTags) * 100).toFixed(2)}%`,
      trimester: "2024 T1",
    }));
    tagPopularityArray.sort((a, b) => parseFloat(b.popularity) - parseFloat(a.popularity));
    return tagPopularityArray;
  };

  const recordsPage = parseInt(pagefilter);
  const lastIndex = currentPage * recordsPage;
  const firstIndex = lastIndex - recordsPage;
  const records = filteredStudies.slice(firstIndex, lastIndex);
  const npage = Math.ceil(filteredStudies.length / recordsPage);

  const tri1 = caseStudies.filter((item) => item.trimester === "2024 T1").length;
  const tri2 = caseStudies.filter((item) => item.trimester === "2").length;
  const tri3 = caseStudies.filter((item) => item.trimester === "3").length;

  const data1 = {
    labels: ["Trimester 1", "Trimester 2", "Trimester 3"],
    datasets: [
      {
        label: "Case Studies",
        backgroundColor: "#3EB470",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        data: [tri1, tri2, tri3],
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { beginAtZero: true } },
  };

  const tags = Array.from(new Set(caseStudies.map((study) => study.tag)));
  const trimesters = Array.from(new Set(caseStudies.map((study) => study.trimester)));

  function prePage() {
    if (currentPage !== 1) setCurrentPage(currentPage - 1);
  }
  function nextPage() {
    if (currentPage !== npage) setCurrentPage(currentPage + 1);
  }

  return (
    <div className="font-sans bg-gray-100 text-black dark:bg-black dark:text-white min-h-screen flex flex-col">
      <Header />
      <h1 className="text-7xl font-bold px-8 pt-4 pb-16">{t("Statistics")}</h1>
      <div className="flex flex-col md:flex-row justify-center gap-10 mb-20">
        <div className="bg-white dark:bg-gray-900 shadow-lg h-auto w-full md:w-[40rem] mb-20 pb-40">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data1} options={options} />
          </div>
        </div>
      </div>

      <main className="w-full flex-1 px-12">
        <section aria-label="Statistics section">
          <div className="mb-4 flex flex-wrap gap-4">
            <select
              value={trimesterFilter}
              onChange={(e) => setTrimesterFilter(e.target.value)}
              className="p-2 border shadow-lg bg-white dark:bg-gray-800 dark:text-white dark:border-gray-600"
            >
              <option value="">{t("All Trimesters")}</option>
              {trimesters.map((trimester) => (
                <option key={trimester} value={trimester}>{trimester}</option>
              ))}
            </select>
            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              className="p-2 border shadow-lg bg-white dark:bg-gray-800 dark:text-white dark:border-gray-600"
            >
              <option value="">{t("All Tags")}</option>
              {tags.map((tag) => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>

          <div className="flex justify-center mb-4">
            <div className="w-full md:w-1/3 bg-white dark:bg-gray-900 shadow-xl py-8 px-10">
              <h2 className="text-2xl font-bold text-gray-400 dark:text-gray-200">{t("Total Results")}</h2>
              <p className="text-3xl font-bold text-center pt-4">{filteredStudies.length}</p>
            </div>
          </div>

          <div className="overflow-hidden p-2 rounded-lg shadow bg-[#3EB470]">
            <form className="flex items-center w-full" style={{color: 'black'}}>
              <input
                type="search"
                placeholder={t("Enter Tag name")}
                className="w-2/5 p-4 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </form>
            <table className="min-w-full bg-white dark:bg-gray-900 text-black dark:text-white">
              <thead>
                <tr>
                  <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider dark:text-gray-100">{t("No")}</th>
                  <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider dark:text-gray-100">{t("Tag")}</th>
                  <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider dark:text-gray-100">{t("number")}</th>
                  <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider dark:text-gray-100">{t("Popularity")}</th>
                </tr>
              </thead>
              <tbody>
                {records.filter((item) => search === "" || item.tag.toLowerCase().includes(search.toLowerCase())).map((study, index) => (
                  <tr key={study.id} className={index % 2 !== 0 ? "bg-[#3EB470]" : "bg-white dark:bg-gray-800"}>
                    <td className="px-5 py-5 border-b border-gray-200 text-sm">{study.id}</td>
                    <td className="px-5 py-5 border-b border-gray-200 text-sm">{study.tag}</td>
                    <td className="px-5 py-5 border-b border-gray-200 text-sm">{study.publishNumber}</td>
                    <td className="px-5 py-5 border-b border-gray-200 text-sm">{study.popularity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <nav className="p-3 mt-5 flex justify-between items-center bg-[#3EB470] text-black dark:text-white">
            <p>{firstIndex + 1} - {lastIndex} of {filteredStudies.length}</p>
            <div className="flex gap-2">
              <button onClick={prePage} disabled={currentPage === 1} className={`font-bold py-1 px-2 ${currentPage === 1 ? "opacity-50" : "hover:underline"}`}>{"<"}</button>
              <span>Page {currentPage}</span>
              <button onClick={nextPage} disabled={currentPage === npage} className={`font-bold py-1 px-2 ${currentPage === npage ? "opacity-50" : "hover:underline"}`}>{">"}</button>
            </div>
            <div className="flex items-center gap-2">
              <span>{t("Rows per page")}</span>
              <select
                value={pagefilter}
                onChange={(e) => setPageFilter(e.target.value)}
                className="p-2 border shadow-lg" style={{color: 'black'}}
              >
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="20">20</option>
                <option value="30">30</option>
              </select>
            </div>
          </nav>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default Statistics;