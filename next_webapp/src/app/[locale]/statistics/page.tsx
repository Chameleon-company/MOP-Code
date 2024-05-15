"use client";
import React, { useState, useEffect, useMemo } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { set } from "firebase/database";
//import caseStudies from './database';  // Adjust the path according to your project structure

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
    {
      id: 1,
      tag: "Safety and Well-being",
      publishNumber: "4",
      popularity: "11%",
      trimester: "1",
    },
    {
      id: 2,
      tag: "Environment and Sustainability",
      publishNumber: "5",
      popularity: "20%",
      trimester: "2",
    },
    {
      id: 3,
      tag: "Business and activity",
      publishNumber: "8",
      popularity: "90%",
      trimester: "3",
    },
    {
      id: 4,
      tag: "Safety and Well-being",
      publishNumber: "4",
      popularity: "11%",
      trimester: "2",
    },
    {
      id: 5,
      tag: "Environment and Sustainability",
      publishNumber: "5",
      popularity: "20%",
      trimester: "3",
    },
    {
      id: 6,
      tag: "Business and activity",
      publishNumber: "8",
      popularity: "90%",
      trimester: "2",
    },
    {
      id: 7,
      tag: "Safety and Well-being",
      publishNumber: "4",
      popularity: "11%",
      trimester: "2",
    },
    {
      id: 8,
      tag: "Environment and Sustainability",
      publishNumber: "5",
      popularity: "20%",
      trimester: "1",
    },
    {
      id: 9,
      tag: "Business and activity",
      publishNumber: "8",
      popularity: "90%",
      trimester: "2",
    },
  ];

  // State for storing the filtered results and all filters
  const [filteredStudies, setFilteredStudies] = useState(caseStudies);
  const [tagFilter, setTagFilter] = useState("");
  const [publishFilter, setPublishFilter] = useState("");
  const [popularityFilter, setPopularityFilter] = useState("");
  const [pagefilter, setPageFilter] = useState("5");
  const [serach, setSearchTerm] = useState("");

  // Distinct tags for the dropdown
  const tags = Array.from(new Set(caseStudies.map((study) => study.tag)));

  // Effect to handle filtering based on tag, publish number, and popularity
  useEffect(() => {
    let filtered = caseStudies;
    if (tagFilter) {
      filtered = filtered.filter((study) => study.tag === tagFilter);
    }
    if (publishFilter) {
      filtered = filtered.filter((study) => {
        const publishNum = parseInt(study.publishNumber, 10);
        switch (publishFilter) {
          case "lessThan4":
            return publishNum < 4;
          case "between4And7":
            return publishNum >= 4 && publishNum <= 7;
          case "above7":
            return publishNum > 7;
          default:
            return true;
        }
      });
    }
    if (popularityFilter) {
      filtered = filtered.filter((study) => {
        const popularity = parseFloat(study.popularity.replace("%", ""));
        switch (popularityFilter) {
          case "0to20":
            return popularity > 0 && popularity <= 20;
          case "20to40":
            return popularity > 20 && popularity <= 40;
          case "40to60":
            return popularity > 40 && popularity <= 60;
          case "60to80":
            return popularity > 60 && popularity <= 80;
          case "80to100":
            return popularity > 80 && popularity <= 100;
          default:
            return true;
        }
      });
    }
    setFilteredStudies(filtered);
  }, [tagFilter, publishFilter, popularityFilter]);

  const [currentPage, setCurrentPage] = useState(1);
  const recordsPage = parseInt(pagefilter);
  const lastindex = currentPage * recordsPage;
  const firstindex = lastindex - recordsPage;
  const records = filteredStudies.slice(firstindex, lastindex);
  const npage = Math.ceil(filteredStudies.length / recordsPage);
  // const numbers = [...Array(npage + 1).keys()].slice(1);

  // Counting the values of trimester to plot on the graph
  const tri1 = caseStudies.filter((item) => item.trimester == "1").length;
  const tri2 = caseStudies.filter((item) => item.trimester == "2").length;
  const tri3 = caseStudies.filter((item) => item.trimester == "3").length;

  // Store the reqired variable for plotting the graph
  const data = {
    labels: ["Trimester 1", "Triemster 2", "Trimester 3"],
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

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        maintainAspectRatio: false,
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
      <div className="flex ">
        <div className="bg-white shadow-2xl ml-[10rem] h-[22rem] w-[40rem] mb-[5rem] pb-[10rem]">
          <h4 className="m-10 font-bold text-[25px]">{t("t1")} </h4>
          <div className="mx-5">
            <Bar data={data} height={"25%"} width={"90%"} options={options} />
          </div>
        </div>
      </div>
      <main style={{ flex: "1 0 auto", width: "100%" }}>
        <div style={{ padding: "0 50px" }}>
          <section aria-label="Statistics section">
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
            <select
              value={publishFilter}
              onChange={(e) => setPublishFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("All Publish Numbers")}</option>
              <option value="lessThan4">{t("Less than 4")}</option>
              <option value="between4And7">{t("Between 4 and 7")}</option>
              <option value="above7">{t("Above 7")}</option>
            </select>
            <select
              value={popularityFilter}
              onChange={(e) => setPopularityFilter(e.target.value)}
              className="p-2 m-2 border shadow-lg"
            >
              <option value="">{t("t2")}</option>
              <option value="0to20">0 - 20%</option>
              <option value="20to40">20 - 40%</option>
              <option value="40to60">40 - 60%</option>
              <option value="60to80">60 - 80%</option>
              <option value="80to100">80 - 100%</option>
            </select>

            <div className="flex">
              <div className="border-solid bg-white shadow-2xl border-2 border-black-600 py-8 px-10 my-10">
                <h2 className=" text-2xl font-bold text-gray-400 ">
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
            <div className="overflow-hidden rounded-lg shadow">
              <form className="flex items-center w-full">
                <input
                  type="search"
                  placeholder={t("Enter Tag name")}
                  className="w-full px-4 py-2 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </form>
              <table className="min-w-full bg-white">
                <thead>
                  <tr>
                    <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg">
                      {t("ID")}
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
                    .filter((item) => {
                      return serach.toLowerCase() === ""
                        ? item
                        : item.tag.toLowerCase().includes(serach);
                    })
                    .map((study, index) => (
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
            <nav className="bg-gray-200 py-3">
              <ul className="pagei">
                <li className="page-item">
                  <button className="  float-right  text-black font-bold py-1 px-2 ">
                    <a href="#" className="page-link" onClick={nextPage}>
                      {">"}
                    </a>
                  </button>
                  <p className="float-right text-black py-1 px-2">
                    {currentPage}
                  </p>
                </li>
                <li className="page-item">
                  <button className=" float-right  text-black font-bold py-1 px-2 ">
                    <a href="#" className="page-link" onClick={prePage}>
                      {"<"}
                    </a>
                  </button>
                </li>

                <li className="float-right py-2">
                  <select
                    value={pagefilter}
                    onChange={(e) => setPageFilter(e.target.value)}
                  >
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="30">30</option>
                  </select>
                </li>
                <li className="float-right">
                  <p className="px-3 py-1">{t("Rows per page")}</p>
                </li>
                <li>
                  <p>
                    {firstindex + 1} - {lastindex} of {filteredStudies.length}
                  </p>
                </li>
              </ul>
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
    } else {
      setCurrentPage(1);
    }
  }

  function nextPage() {
    if (currentPage !== npage) {
      setCurrentPage(currentPage + 1);
    } else {
      setCurrentPage(npage);
    }
  }
};

export default Statistics;
