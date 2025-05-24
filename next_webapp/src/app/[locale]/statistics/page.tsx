"use client";
// import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
// import {
//   BarElement,
//   CategoryScale,
//   Chart as ChartJS,
//   Legend,
//   LinearScale,
//   Title,
//   Tooltip,
// } from "chart.js";
// import { useTranslations } from "next-intl";
// import { useEffect, useState } from "react";
// import { Bar } from "react-chartjs-2";
// import Footer from "../../../components/Footer";
// import Header from "../../../components/Header";

// ChartJS.register(
//   CategoryScale,
//   LinearScale,
//   BarElement,
//   Title,
//   Tooltip,
//   Legend
// );

// const Statistics = () => {

//   //Dummy array
//   const dummyArray = [{ id: 1, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "1" },
//   { id: 2, tag: "Environment and Sustainability", publishNumber: "5", popularity: "40%", trimester: "2" },
//   { id: 3, tag: "Business and activity", publishNumber: "8", popularity: "90%", trimester: "3" },
//   { id: 4, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
//   { id: 5, tag: "Environment and Sustainability", publishNumber: "5", popularity: "90%", trimester: "3" },
//   { id: 6, tag: "Business and activity", publishNumber: "8", popularity: "70%", trimester: "1" },
//   { id: 7, tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
//   { id: 8, tag: "Environment and Sustainability", publishNumber: "5", popularity: "20%", trimester: "2" },
//   { id: 9, tag: "Business and activity", publishNumber: "8", popularity: "60%", trimester: "1" },]

//   //Stats array
//   const [caseStudies, setStats] = useState([
//   ])

//   //Importing case studies
//   useEffect(() => {
//     searchUseCases({ searchTerm: "", searchMode: SEARCH_MODE.TITLE, category: CATEGORY.ALL }).then((useCases) => {
//       const stats = getStats(useCases.filteredStudies)
//       setStats(stats)
//     })

//   }, [])
//   async function searchUseCases(searchParams: SearchParams) {
//     const response = await fetch("/api/search-use-cases", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify(searchParams),
//     });

//     if (!response.ok) {
//       throw new Error(`HTTP error! status: ${response.status}`);
//     }

//     return await response.json();
//   }
//   //Get stats
//   const getStats = (caseStudiesArray: CaseStudy[]) => {
//     const tagCountMap: { [key: string]: number } = {};
//     let totalTags = 0;

//     // Count occurrences of each tag across all case studies
//     caseStudiesArray.forEach(caseStudy => {
//       caseStudy.tags.forEach(tag => {
//         const normalizedTag = tag.trim().toLowerCase(); // Normalize tags for consistency
//         if (!tagCountMap[normalizedTag]) {
//           tagCountMap[normalizedTag] = 0;
//         }
//         tagCountMap[normalizedTag]++;
//         totalTags++;
//       });
//     });

//     // Calculate popularity
//     const tagPopularityArray = Object.keys(tagCountMap).map(tag => ({
//       tag: tag,
//       publishNumber: tagCountMap[tag],
//       popularity: `${((tagCountMap[tag] / totalTags) * 100).toFixed(2)}%`,
//       trimester: "2024 T1"
//     }));

//     // Sort by popularity
//     tagPopularityArray.sort((a, b) => {
//       const popularityA = parseFloat(a.popularity.replace('%', ''));
//       const popularityB = parseFloat(b.popularity.replace('%', ''));
//       return popularityB - popularityA; // Descending order
//     });

//     // Assign IDs after sorting
//     tagPopularityArray.forEach((item, index) => {
//       item.id = index + 1;
//     });

//     return tagPopularityArray;
//   };


//   // State for storing the filtered results and all filters
//   const [filteredStudies, setFilteredStudies] = useState(caseStudies);
//   const [tagFilter, setTagFilter] = useState("");
//   const [trimesterFilter, setTrimesterFilter] = useState("");
//   const [pagefilter, setPageFilter] = useState("5");
//   const [search, setSearchTerm] = useState("");

//   // Distinct tags for the dropdown
//   const tags = Array.from(new Set(caseStudies.map((study) => study.tag)));
//   const trimesters = Array.from(new Set(caseStudies.map((study) => study.trimester)));

//   // Effect to handle filtering based on tag and trimester



//New code after intergraipn the db if you want to see the hard code one comment following and uncomment above

import { CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
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
  const t = useTranslations("statistics");
  
  // States for API data
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: "Number of Case Studies",
        backgroundColor: "#3EB470",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        data: [],
      },
    ],
  });
  
  const [totalResults, setTotalResults] = useState(0);
  const [tagsData, setTagsData] = useState([]);
  const [totalPages, setTotalPages] = useState(1);
  
  // State for storing the filtered results and all filters
  const [tagFilter, setTagFilter] = useState("");
  const [trimesterFilter, setTrimesterFilter] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState("5");
  const [searchTerm, setSearchTerm] = useState("");

  // Fetch available tags and trimesters for filters
  const [availableTags, setAvailableTags] = useState([]);
  const [availableTrimesters, setAvailableTrimesters] = useState([]);

  // Fetch trimester chart data
  useEffect(() => {
    fetch("/api/statistics/trimester")
      .then((response) => response.json())
      .then((data) => {
        setChartData({
          labels: data.labels,
          datasets: [
            {
              label: "Number of Case Studies",
              backgroundColor: "#3EB470",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 1,
              data: data.data,
            },
          ],
        });
        
        // Extract trimesters for filter dropdown
        setAvailableTrimesters(data.labels);
      })
      .catch((error) => console.error("Error fetching chart data:", error));
  }, []);

  // Fetch total results count based on filters
  useEffect(() => {
    let url = "/api/statistics/count";
    const params = [];
    
    if (trimesterFilter) {
      // Format the trimester value properly
      // For example, if the value is "Trimester 1", convert it to "2024-T1"
      const trimNumber = trimesterFilter.replace("Trimester ", "");
      params.push(`trimester=2024-T${trimNumber}`);
    }
    
    if (tagFilter) {
      params.push(`tag=${encodeURIComponent(tagFilter)}`);
    }
    
    if (params.length > 0) {
      url += `?${params.join("&")}`;
    }
    
    console.log("Fetching count from:", url);
    
    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        console.log("Count response:", data);
        setTotalResults(data.total);
      })
      .catch((error) => console.error("Error fetching count:", error));
  }, [trimesterFilter, tagFilter]);

  // Fetch tags data for table
  useEffect(() => {
    let url = `/api/statistics/tags?page=${currentPage}&limit=${pageSize}`;
    
    if (searchTerm) {
      url += `&search=${encodeURIComponent(searchTerm)}`;
    }
    
    if (tagFilter) {
      url += `&tag=${encodeURIComponent(tagFilter)}`;
    }
    
    if (trimesterFilter) {
      // Format the trimester value properly
      const trimNumber = trimesterFilter.replace("Trimester ", "");
      url += `&trimester=2024-T${trimNumber}`;
    }
    
    console.log("Fetching tags from:", url);
    
    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        console.log("Tags response:", data);
        setTagsData(data.data);
        setTotalPages(data.pagination.totalPages);
        
        // Extract unique tags for filter dropdown if not already populated
        if (availableTags.length === 0) {
          // Fetch all available tags for the dropdown
          fetch("/api/statistics/tags?page=1&limit=100")
            .then(response => response.json())
            .then(allData => {
              const uniqueTags = Array.from(new Set(allData.data.map(item => item.tag)));
              setAvailableTags(uniqueTags);
            })
            .catch(error => console.error("Error fetching all tags:", error));
        }
      })
      .catch((error) => console.error("Error fetching tags data:", error));
  }, [currentPage, pageSize, searchTerm, tagFilter, trimesterFilter]);

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

  function prePage() {
    if (currentPage !== 1) {
      setCurrentPage(currentPage - 1);
    }
  }

  function nextPage() {
    if (currentPage !== totalPages) {
      setCurrentPage(currentPage + 1);
    }
  }

  // Calculate pagination info
  const firstIndex = ((currentPage - 1) * parseInt(pageSize)) + 1;
  const lastIndex = Math.min(currentPage * parseInt(pageSize), totalResults);

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
        {t("Statistics")}
      </h1>

      {/* Chart container */}
      <div className="flex flex-col md:flex-row justify-center gap-10 mb-[5rem]">
        <div className="bg-white shadow-l h-auto w-full md:w-[40rem] mb-[5rem] pb-[10rem]">
          <h4 className="m-10 font-bold text-[15px]">{t("t1")}</h4>
          <div className="mx-5">
            <Bar data={chartData} options={options} />
          </div>
        </div>
      </div>

      <main style={{ flex: "1 0 auto", width: "100%" }}>
        <div style={{ padding: "0 50px" }}>
          <section aria-label="Statistics section">
            {/* Filter Dropdowns */}
            <select
              value={trimesterFilter}
              onChange={(e) => {
                setTrimesterFilter(e.target.value);
                setCurrentPage(1); // Reset to first page when filter changes
              }}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("All Trimesters")}</option>
              <option value="Trimester 1">Trimester 1</option>
              <option value="Trimester 2">Trimester 2</option>
              <option value="Trimester 3">Trimester 3</option>
            </select>

            <select
              value={tagFilter}
              onChange={(e) => {
                setTagFilter(e.target.value);
                setCurrentPage(1); // Reset to first page when filter changes
              }}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("All Tags")}</option>
              {availableTags.map((tag) => (
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
                  {totalResults}
                </p>
              </div>
            </div>
            
            {/* Search and Table */}
            <div className="overflow-hidden p-2 rounded-lg shadow bg-[#3EB470]">
              <form className="flex items-center w-full" onSubmit={(e) => e.preventDefault()}>
                <input
                  type="search"
                  placeholder={t("Enter Tag name")}
                  className="w-2/5 p-4 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1); // Reset to first page when searching
                  }}
                  value={searchTerm}
                />
              </form>
              <table className="min-w-full bg-white">
                <thead>
                  <tr>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("No")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("Tag")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("number")}
                    </th>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-[#3EB470] text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      {t("Popularity")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tagsData.map((item, index) => (
                    <tr
                      key={item.no}
                      className={index % 2 !== 0 ? "bg-[#3EB470]" : "bg-white"} // Every other row green
                    >
                      <td className="px-5 py-5 border-b border-gray-200 text-sm">
                        {item.no}
                      </td>
                      <td className="px-5 py-5 border-b border-gray-200 text-sm">
                        {item.tag}
                      </td>
                      <td className="px-5 py-5 border-b border-gray-200 text-sm">
                        {item.numberOfTestCasesPublished}
                      </td>
                      <td className="px-5 py-5 border-b border-gray-200 text-sm">
                        {item.popularity}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Pagination */}
            <nav className="p-3 mt-5 flex justify-between items-center bg-[#3EB470]">
              <p>
                {firstIndex} - {lastIndex} of {totalResults}
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
                  className={`text-black font-bold py-1 px-2 ${currentPage === totalPages ? "cursor-not-allowed opacity-50" : ""}`}
                  onClick={nextPage}
                  disabled={currentPage === totalPages}
                >
                  {">"}
                </button>
              </div>
              <div className="flex justify-between">
                <p className="py-1">{t("Rows per page")}</p>
                <select
                  value={pageSize}
                  onChange={(e) => {
                    setPageSize(e.target.value);
                    setCurrentPage(1); // Reset to first page when changing page size
                  }}
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
};

export default Statistics;