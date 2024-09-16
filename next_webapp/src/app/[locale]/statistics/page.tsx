"use client";
import React, {useEffect, useState} from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import {useTranslations} from "next-intl";
import {BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title, Tooltip,} from "chart.js";
import {Bar} from "react-chartjs-2";
import {CATEGORY, SEARCH_MODE, SearchParams} from "@/app/types";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

async function searchStatistics(searchParams: SearchParams) {
  const response = await fetch("/api/statistics", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    }
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}


const Statistics = () => {
  const [caseStudies, setCaseStudies] = useState([]);
  // State for storing the filtered results and all filters
  const [filteredStudies, setFilteredStudies] = useState(caseStudies);

  useEffect(() => {
    fetchStatistics()
  }, [])

  const fetchStatistics = async () => {
    try {
      const data = await searchStatistics({searchTerm: "",
        searchMode: SEARCH_MODE.TITLE,
        category: CATEGORY.EV});
      setCaseStudies(data);
      console.log('data received! ' + data.toString());
    } catch (error) {
      console.log('request called!');
    } finally {
      console.log('request called finally!');
    }
  };

  const [tagFilter, setTagFilter] = useState("");
  const [trimesterFilter, setTrimesterFilter] = useState("");
  const [pagefilter, setPageFilter] = useState("5");
  const [search, setSearchTerm] = useState("");

  // Effect to handle filtering based on tag and trimester
  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) {
      filtered = filtered.filter((study: {tag: string}) => study.tag === tagFilter);
    }
    if (trimesterFilter) {
      filtered = filtered.filter((study: {trimester: string}) => study.trimester === trimesterFilter);
    }
    setFilteredStudies(filtered);
  }, [tagFilter, trimesterFilter]);

  // Distinct tags for the dropdown
  const tags = Array.from(new Set(caseStudies.map((study: {tag: string}) => study.tag)));
  // Distinct trimesters for the dropdown
  const trimesters = Array.from(new Set(caseStudies.map((study: {trimester: string}) => study.trimester)));

  // Effect to handle filtering based on tag and trimester
  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) {
      filtered = filtered.filter((study: {tag: string}) => study.tag === tagFilter);
    }
    if (trimesterFilter) {
      filtered = filtered.filter((study: {trimester: string}) => study.trimester === trimesterFilter);
    }
    setFilteredStudies(filtered);
  }, [tagFilter, trimesterFilter]);

  const [currentPage, setCurrentPage] = useState(1);
  const recordsPage = parseInt(pagefilter);
  const lastIndex = currentPage * recordsPage;
  const firstIndex = lastIndex - recordsPage;
  const records = filteredStudies.slice(firstIndex, lastIndex);
  const npage = Math.ceil(filteredStudies.length / recordsPage);

  // Sort caseStudies by popularity
  const sortedStudiesByPopularity = [...caseStudies].sort((a: {popularity: string}, b: {popularity: string}) => {
    const popularityA = parseFloat(a.popularity.replace('%', ''));
    const popularityB = parseFloat(b.popularity.replace('%', ''));
    return popularityB - popularityA; // Descending order
  });


  // Counting the values of trimester to plot on the graph
  const tri1 = caseStudies.filter((item: {trimester: string}) => item.trimester === "1").length;
  const tri2 = caseStudies.filter((item: {trimester: string}) => item.trimester === "2").length;
  const tri3 = caseStudies.filter((item: {trimester: string}) => item.trimester === "3").length;

  // Store the required variables for plotting the graph
  const data1 = {
    labels: ["Trimester 1", "Trimester 2", "Trimester 3"],
    datasets: [
      {
        label: "Data Series 1",
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        data: [tri1, tri2, tri3],
      },
    ],
  };

  const data2 = {
    labels: ["Trimester 1", "Trimester 2", "Trimester 3"],
    datasets: [
      {
        label: "Data Series 2",
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        data: [tri1, tri2, tri3],
      },
    ],
  };

  // Responsive chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const t = useTranslations("statistics");

  return (
    <div
      style={{
        margin: "0 auto",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
      className="font-sans bg-gray-100 text-black"
      role="main"
      aria-label="Statistics page"
    >
      <Header />
      <h1 className="text-7xl font-bold px-[2rem] pt-[1rem] pb-[4rem]">
      {" "}
      {t("Statistics")}{" "}
      </h1>

      {/* Flex container for charts */}
      <div className="flex flex-col md:flex-row justify-center gap-10 mb-[5rem]">
        <div className="bg-white shadow-l h-auto w-full md:w-[40rem] mb-[5rem] pb-[10rem]">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data1} options={options} />
          </div>
        </div>
        <div className="bg-white shadow-l h-auto w-full md:w-[40rem] mb-[5rem] pb-[10rem]">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data2} options={options} />
          </div>
        </div>
      </div>

      <main style={{ flex: "1 0 auto", width: "100%" }}>
        <div style={{ padding: "0 50px" }}>
          <section aria-label="Statistics section">

            {/* Filter Dropdowns */}
            <select
              value={trimesterFilter}
              onChange={(e) => setTrimesterFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("All Trimesters")}</option>
              {trimesters.map((trimester) => (
                <option key={trimester} value={trimester}>
                  {t(`Trimester${trimester}`)}
                </option>
              ))}
            </select>

            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("All Tags")}</option>
              {tags.map((tag) => (
                <option key={tag} value={tag}>
                  {tag}
                </option>
              ))}
            </select>

            {/* Total Results */}
            <div className="flex justify-center mb-4">
              <div className="w-full md:w-1/3 bg-white shadow-xl py-8 px-10">
                <h2 className="text-2xl font-bold text-gray-400">
                  {t("Total Results")}
                </h2>
                <p className="text-[1.8rem] font-bold text-center pt-[15px] text-black-400">
                  {filteredStudies.length}
                </p>
              </div>
            </div>

            {/* Search Input */}
            <div className="overflow-hidden p-2 rounded-lg shadow">
              <form className="flex items-center w-full">
                <input
                  type="search"
                  placeholder={t("Enter Tag name")}
                  className="w-2/5 p-4 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </form>

              {/* Data Table */}
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white">
                  <thead>
                    <tr>
                      <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg">
                        {t("No")}
                      </th>
                      <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg">
                        {t("Tag")}
                      </th>
                      <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                        {t("number")}
                      </th>
                      <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tr-lg">
                        {t("Popularity")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {records
                      .filter((item: {tag: string}) => {
                        return search.toLowerCase() === ""
                          ? item
                          : item.tag.toLowerCase().includes(search);
                      })
                      .map((study: {id: string, tag: string, publishNumber: string, popularity: string }, index) => (
                        <tr
                          key={study.id}
                          className={index % 2 === 0 ? "bg-gray-100" : "bg-white"}
                        >
                          <td className="px-5 py-5 border-b border-gray-200 text-sm">
                            {study.id}
                          </td>
                          <td className="px-5 py-5 border-b border-gray-200 text-sm">
                            {study.tag}
                          </td>
                          <td className="px-5 py-5 border-b border-gray-200 text-sm">
                            {study.publishNumber}
                          </td>
                          <td className="px-5 py-5 border-b border-gray-200 text-sm">
                            {study.popularity}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pagination */}
            <nav className="bg-gray-200 p-3 mt-5 flex justify-between items-center">
              <p>
                {firstIndex + 1} - {lastIndex} of {filteredStudies.length}
              </p>
              <div className="flex justify-between">
                <button
                  className={`text-black font-bold py-1 px-2 ${currentPage === 1 ? "cursor-not-allowed opacity-50" : ""}`}
                  onClick={prePage}
                  disabled={currentPage === 1}
                >
                  {"<"}
                </button>
                <p className="text-black py-1 px-2">Page {currentPage}</p>
                <button
                  className={`text-black font-bold py-1 px-2 ${currentPage === npage ? "cursor-not-allowed opacity-50" : ""}`}
                  onClick={nextPage}
                  disabled={currentPage === npage}
                >
                  {">"}
                </button>
              </div>
              <div className="flex justify-between">
                <p className="py-1">{t("Rows per page")}</p>
                <select
                  value={pagefilter}
                  onChange={(e) => setPageFilter(e.target.value)}
                  className="p-2 border shadow-lg"
                >
                  <option value="5">5</option>
                  <option value="10">10</option>
                  <option value="20">20</option>
                  <option value="30">30</option>
                </select>
              </div>
            </nav>
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );

  function prePage() {
    if (currentPage !== 1) {
      setCurrentPage(currentPage - 1);
    }
  }

  function nextPage() {
    if (currentPage !== npage) {
      setCurrentPage(currentPage + 1);
    }
  }
};

export default Statistics;
