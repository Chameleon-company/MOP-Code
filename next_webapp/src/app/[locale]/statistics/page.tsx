"use client";
import React, { useState, useEffect } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import Tooglebutton from "../Tooglebutton/Tooglebutton";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Statistics = () => {
  // Dummy Array
  const caseStudies = [
    { id: 1, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "1" },
    { id: 2, tag: "Environment and Sustainability", publishNumber: "5", popularity: "40%", trimester: "2" },
    { id: 3, tag: "Business and activity", publishNumber: "8", popularity: "90%", trimester: "3" },
    { id: 4, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
    { id: 5, tag: "Environment and Sustainability", publishNumber: "5", popularity: "90%", trimester: "3" },
    { id: 6, tag: "Business and activity", publishNumber: "8", popularity: "70%", trimester: "1" },
    { id: 7, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
    { id: 8, tag: "Environment and Sustainability", publishNumber: "5", popularity: "20%", trimester: "2" },
    { id: 9, tag: "Business and activity", publishNumber: "8", popularity: "60%", trimester: "1" },
  ];

  // State for storing the filtered results and all filters
  const [filteredStudies, setFilteredStudies] = useState(caseStudies);
  const [tagFilter, setTagFilter] = useState("");
  const [trimesterFilter, setTrimesterFilter] = useState("");
  const [pagefilter, setPageFilter] = useState("5");
  const [search, setSearchTerm] = useState("");

  // Distinct tags for the dropdown
  const tags = Array.from(new Set(caseStudies.map((study) => study.tag)));
  const trimesters = Array.from(new Set(caseStudies.map((study) => study.trimester)));

  // Effect to handle filtering based on tag and trimester
  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) {
      filtered = filtered.filter((study) => study.tag === tagFilter);
    }
    if (trimesterFilter) {
      filtered = filtered.filter((study) => study.trimester === trimesterFilter);
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
  const sortedStudiesByPopularity = [...caseStudies].sort((a, b) => {
    const popularityA = parseFloat(a.popularity.replace('%', ''));
    const popularityB = parseFloat(b.popularity.replace('%', ''));
    return popularityB - popularityA; // Descending order
  });


  // Counting the values of trimester to plot on the graph
  const tri1 = caseStudies.filter((item) => item.trimester === "1").length;
  const tri2 = caseStudies.filter((item) => item.trimester === "2").length;
  const tri3 = caseStudies.filter((item) => item.trimester === "3").length;

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

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        maintainAspectRatio: false,
      },
    },
  };

   //dark theme
   const [dark_value,setdarkvalue] = useState(false);
  
   const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean))=>{
     setdarkvalue(newValue);
   }

  const t = useTranslations("statistics");

  return (
    <div className= {`${dark_value && "dark"}`}>
    <div
      style={{
        margin: "0 auto",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
      className="font-sans bg-gray-100 text-black dark:bg-pr_bg_dark dark:text-white"
      role="main"
      aria-label="Statistics page"
    >
      <Header />
      <Tooglebutton onValueChange={handleValueChange}/>
      <h1 className="text-7xl font-bold px-[2rem] pt-[1rem] pb-[4rem]">
      {" "}
      {t("Statistics")}{" "}
      </h1>
      <div className="flex justify-center gap-20 mb-[5rem]">
        <div className="bg-white shadow-l h-[30rem] w-[40rem] mb-[5rem] pb-[10rem] dark:bg-sc_bg_dark">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data1} height={15} width={25} options={options} />
          </div>
        </div>
        <div className="bg-white shadow-l h-[30rem] w-[40rem] mb-[5rem] pb-[10rem] dark:bg-sc_bg_dark">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data2} height={15} width={25} options={options} />
          </div>
        </div>
      </div>
      <main style={{ flex: "1 0 auto", width: "100%" }}>
        <div style={{ padding: "0 50px" }}>
          <section aria-label="Statistics section">
          <select
              value={trimesterFilter}
              onChange={(e) => setTrimesterFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg dark:bg-sc_bg_dark dark:border-0 dark:shadow-[#454242]"
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
              className="p-2 m-2 border shadow-lg dark:bg-sc_bg_dark dark:border-0 dark:shadow-[#454242]"
            >
              <option value="">{t("All Tags")}</option>
              {tags.map((tag) => (
                <option key={tag} value={tag}>
                  {tag}
                </option>
              ))}
            </select>
            <div className="flex">
              <div className="border-solid bg-white shadow-2xl border-2 border-black-600 py-8 px-10 my-10 dark:bg-sc_bg_dark dark:border-0 dark:shadow-[#454242]">
                <h2 className="text-2xl font-bold text-gray-400">
                  {t("Total Results")}
                </h2>
                <p className="text-[1.8rem] font-bold text-center pt-[15px] px-2rem font-bold text-black-400 ">
                  {filteredStudies.length}
                </p>
              </div>
              <div>
                <p></p>
              </div>
            </div>
            <div className="overflow-hidden p-2 rounded-lg shadow ">
              <form className="flex items-center w-full ">
                <input
                  type="search"
                  placeholder={t("Enter Tag name")}
                  className="w-2/5 p-4 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500 dark:bg-sc_bg_dark  dark:text-white"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </form>
              <table className="min-w-full bg-white dark:bg-sc_bg_dark dark:border-0">
                <thead>
                  <tr>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg dark:bg-sc_bg_dark dark:border-0 dark:text-white">
                      {t("No")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg dark:bg-sc_bg_dark dark:border-0 dark:text-white">
                      {t("Tag")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider dark:bg-sc_bg_dark dark:border-0 dark:text-white">
                      {t("number")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tr-lg dark:bg-sc_bg_dark dark:border-0 dark:text-white">
                      {t("Popularity")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {records
                    .filter((item) => {
                      return search.toLowerCase() === ""
                        ? item
                        : item.tag.toLowerCase().includes(search);
                    })
                    .map((study, index) => (
                      <tr
                        key={study.id}
                        className={index % 2 === 0 ? "bg-gray-100 dark:bg-sc_bg_dark  dark:text-white" : "bg-white dark:bg-[#6B6A6A]"}
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
            <nav className="bg-gray-200 p-3 mt-5 flex justify-between items-center dark:bg-sc_bg_dark">
              <p>
                {firstIndex + 1} - {lastIndex} of {filteredStudies.length}
              </p>
              <div className="flex justify-between">
                <button
                  className={`text-black font-bold py-1 px-2 dark:text-white ${currentPage === 1 ? "cursor-not-allowed opacity-50" : ""}`}
                  onClick={prePage}
                  disabled={currentPage === 1}
                >
                  {"<"}
                </button>
                <p className="text-black py-1 px-2 dark:text-white">Page {currentPage}</p>
                <button
                  className={`text-black font-bold py-1 dark:text-white px-2 ${currentPage === npage ? "cursor-not-allowed opacity-50" : ""}`}
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
                  className="p-2 border shadow-lg dark:bg-pr_bg_dark dark:border-0"
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