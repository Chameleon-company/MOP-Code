"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { HiMoon, HiSun } from "react-icons/hi2";
import "../../../../public/styles/privacy.css";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");

  const [searchTerm, setSearchTerm] = useState<string>("");
  const [openSections, setOpenSections] = useState<{ [key: string]: boolean }>({});
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleTheme = () => {
    setIsDarkMode((prev) => !prev);
  };

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    if (stored === "dark") setIsDarkMode(true);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (isDarkMode) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  const sections = [
    { key: "1", title: t("t1"), content: t("p1") },
    { key: "2", title: t("t2"), content: t("p2") },
    { key: "3", title: t("t3"), content: t("p3") },
    { key: "4", title: t("t4"), content: t("p4") },
    { key: "5", title: t("t5"), content: t("p5") },
    { key: "6", title: t("t6"), content: t("p6") },
  ];

  const filteredSections = sections.filter((section) =>
    section.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const expandAll = () => {
    const expanded: { [key: string]: boolean } = {};
    filteredSections.forEach((section) => {
      expanded[section.key] = true;
    });
    setOpenSections(expanded);
  };

  const collapseAll = () => {
    setOpenSections({});
  };

  const downloadPDF = () => {
    const input = document.querySelector(".policy-box");
    if (!input) return;

    html2canvas(input as HTMLElement).then((canvas: HTMLCanvasElement) => {
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save("privacy-policy.pdf");
    });
  };

  return (
    <div className="flex flex-col min-h-screen bg-white text-gray-900 dark:bg-black dark:text-white transition-colors duration-300">
      <Header />

      <main className="flex-grow flex flex-col items-center font-montserrat relative pb-20 policy-box">
        <h1 className="text-3xl font-bold mt-10 mb-6">{t("Privacy Policy")}</h1>

        <div className="flex flex-col items-center gap-4 w-full max-w-4xl px-4">
          <input
            type="text"
            placeholder="Search sections..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded dark:bg-gray-800 dark:border-gray-600"
          />

          <div className="flex gap-4 flex-wrap justify-center">
            <button onClick={expandAll} className="bg-green-600 text-white px-4 py-2 rounded">
              Expand All
            </button>
            <button onClick={collapseAll} className="bg-green-600 text-white px-4 py-2 rounded">
              Collapse All
            </button>
            <button onClick={downloadPDF} className="bg-green-600 text-white px-4 py-2 rounded">
              Download PDF
            </button>
          </div>

          <div className="w-full mt-6">
            {filteredSections.length === 0 ? (
              <p className="text-center">No matching sections found.</p>
            ) : (
              filteredSections.map(({ key, title, content }) => (
                <div key={key} className="mb-2">
                  <button
                    onClick={() => toggleSection(key)}
                    className="w-full flex justify-between items-center font-bold px-4 py-3 rounded-sm transition bg-[#2ECC71] text-black hover:bg-[#2abb67] dark:bg-[#2ECC71] dark:hover:bg-[#2abb67]"
                  >
                    <span>{title}</span>
                    <span>{openSections[key] ? "▲" : "▼"}</span>
                  </button>
                  {openSections[key] && (
                    <div className="p-4 text-sm rounded-b-sm bg-green-200 text-black dark:bg-[#acecc7]">
                      {content}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="flex items-center justify-center mt-10">
            <p className="text-center text-[14px] max-w-4xl">{t("p7")}</p>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Privacypolicy;
