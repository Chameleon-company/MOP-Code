"use client";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from "chart.js";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import { color } from "chart.js/helpers";
import Footer from "../../../components/Footer";
import Header from "../../../components/Header";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Statistics = () => {

  //Dummy array
  const dummyArray = [{ id: 1, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "1" },
  { id: 2, tag: "Environment and Sustainability", publishNumber: "5", popularity: "40%", trimester: "2" },
  { id: 3, tag: "Business and activity", publishNumber: "8", popularity: "90%", trimester: "3" },
  { id: 4, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
  { id: 5, tag: "Environment and Sustainability", publishNumber: "5", popularity: "90%", trimester: "3" },
  { id: 6, tag: "Business and activity", publishNumber: "8", popularity: "70%", trimester: "1" },
  { id: 7, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
  { id: 8, tag: "Environment and Sustainability", publishNumber: "5", popularity: "20%", trimester: "2" },
  { id: 9, tag: "Business and activity", publishNumber: "8", popularity: "60%", trimester: "1" },]

  //Stats array
  const [caseStudies, setStats] = useState([
  ])

  //Importing case studies
  useEffect(() => {
    searchUseCases({ searchTerm: "", searchMode: SEARCH_MODE.TITLE, category: CATEGORY.ALL }).then((useCases) => {
      const stats = getStats(useCases.filteredStudies)
      setStats(stats)
    })

  }, [])
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
  //Get stats
  const getStats = (caseStudiesArray: CaseStudy[]) => {
    const tagCountMap: { [key: string]: number } = {};
    let totalTags = 0;

    // Count occurrences of each tag across all case studies
    caseStudiesArray.forEach(caseStudy => {
      caseStudy.tags.forEach(tag => {
        const normalizedTag = tag.trim().toLowerCase(); // Normalize tags for consistency
        if (!tagCountMap[normalizedTag]) {
          tagCountMap[normalizedTag] = 0;
        }
        tagCountMap[normalizedTag]++;
        totalTags++;
      });
    });

    // Calculate popularity
    const tagPopularityArray = Object.keys(tagCountMap).map(tag => ({
      tag: tag,
      publishNumber: tagCountMap[tag],
      popularity: `${((tagCountMap[tag] / totalTags) * 100).toFixed(2)}%`,
      trimester: "2024 T1"
    }));

    // Sort by popularity
    tagPopularityArray.sort((a, b) => {
      const popularityA = parseFloat(a.popularity.replace('%', ''));
      const popularityB = parseFloat(b.popularity.replace('%', ''));
      return popularityB - popularityA; // Descending order
    });

    // Assign IDs after sorting
    tagPopularityArray.forEach((item, index) => {
      item.id = index + 1;
    });

    return tagPopularityArray;
  };


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
  const records = caseStudies.slice(firstIndex, lastIndex);
  const npage = Math.ceil(caseStudies.length / recordsPage);


  // Counting the values of trimester to plot on the graph
  const tri1 = caseStudies.filter((item) => item.trimester === "2024 T1").length;
  const tri2 = caseStudies.filter((item) => item.trimester === "2").length;
  const tri3 = caseStudies.filter((item) => item.trimester === "3").length;

  // Store the required variables for plotting the graph
  const data1 = {
    labels: ["Trimester 1", "Trimester 2", "Trimester 3"],
    datasets: [
      {
        label: "Data Series 1",
        backgroundColor: "#3EB470",
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
      className="font-sans bg-gray-100 text-black dark:bg-zinc-800 "
      role="main"
      aria-label="Statistics page"
    >
      <Header />
      <h1 className="text-7xl font-bold px-[2rem] pt-[1rem] pb-[4rem] dark:text-slate-100">
      {" "}
      {t("Statistics")}{" "}

      </h1>

      {/* Flex container for charts */}
      <div className="flex flex-col md:flex-row justify-center gap-10 mb-[5rem] dark:text-slate-100">
        <div className="bg-white dark:bg-zinc-700 shadow-l h-auto w-full md:w-[40rem] mb-[5rem] pb-[10rem]">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data1} options={options} />
          </div>
        </div>
        <div className="bg-white  dark:bg-zinc-700 shadow-l h-auto w-full md:w-[40rem] mb-[5rem] pb-[10rem] dark:text-slate-100">

        {/* <div className="bg-white shadow-l h-[30rem] w-[40rem] mb-[5rem] pb-[10rem]">

          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={data2} options={options} />
          </div>
        </div> */}
      </div>

      <main style={{ flex: "1 0 auto", width: "100%" }}>
        <div style={{ padding: "0 50px" }}>
          <section aria-label="Statistics section">
            {/* Filter Dropdowns */}
            <select
              value={trimesterFilter}
              onChange={(e) => setTrimesterFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg  dark:bg-zinc-700 dark:text-slate-100"
            >
              <option value="">{t("All Trimesters")}</option>
              {trimesters.map((trimester) => (
                <option key={trimester} value={trimester}>
                  {`${trimester}`}
                </option>
              ))}
            </select>

            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg  dark:bg-zinc-700 dark:text-slate-100"
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
              <div className="w-full md:w-1/3 bg-white dark:text-slate-100  dark:bg-zinc-700 shadow-xl py-8 px-10">
                <h2 className="text-2xl font-bold text-gray-400 dark:text-slate-100">
                  {t("Total Results")}
                </h2>
                <p className="text-[1.8rem] font-bold text-center pt-[15px] text-black-400">
                  {filteredStudies.length}
                </p>
              </div>
            </div>
            <div className="overflow-hidden p-2 rounded-lg shadow bg-[#3EB470]">
              <form className="flex items-center w-full">
                <input
                  type="search"
                  placeholder={t("Enter Tag name")}
                  className="w-2/5 p-4 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </form>
              <table className="min-w-full bg-white">
                <thead>
                  <tr>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("No")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470]  text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("Tag")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470]  text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("number")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470]  text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
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
                        className={index % 2 != 0 ? "bg-[#3EB470] " : "bg-white"} // Every other row green
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
            <nav className="p-3 mt-5 flex justify-between items-center bg-[#3EB470]">
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
