"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { HiMoon, HiSun } from "react-icons/hi2";
import "../../../../public/styles/privacy.css";

interface AccordionProps {
  title: string;
  content: string;
  isOpen: boolean;
  onClick: () => void;
}

const AccordionItem: React.FC<AccordionProps> = ({ title, content, isOpen, onClick }) => (
  <div className="accordion bg-white border rounded-lg shadow-md transition-all duration-300 mb-4">
    <button
      onClick={onClick}
      className="w-full text-left p-4 font-bold text-green-600 flex justify-between items-center"
    >
      {title}
      <i className={`fas fa-chevron-${isOpen ? 'up' : 'down'}`} />
    </button>
    <div
      className="accordion-content bg-green-50 overflow-hidden transition-all duration-500 ease-in-out"
      style={{
        maxHeight: isOpen ? '200px' : '0',
        opacity: isOpen ? 1 : 0,
        padding: isOpen ? '1rem' : '0 1rem'
      }}
    >
      <p>{content}</p>
    </div>
  </div>
);

const sections = [
  { titleKey: 't1', contentKey: 'p1' },
  { titleKey: 't2', contentKey: 'p2' },
  { titleKey: 't3', contentKey: 'p3' },
  { titleKey: 't4', contentKey: 'p4' },
  { titleKey: 't5', contentKey: 'p5' },
  { titleKey: 't6', contentKey: 'p6' }
];

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");
  const [openIndexes, setOpenIndexes] = useState<number[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>("");

  const toggleAccordion = (index: number) => {
    setOpenIndexes(prev =>
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const filteredSections = sections.filter((section) =>
    t(section.titleKey).toLowerCase().includes(searchTerm.toLowerCase())
  );

  const expandAll = () => setOpenIndexes(filteredSections.map((_, index) => index));
  const collapseAll = () => setOpenIndexes([]);

  const downloadPDF = () => {
    const input = document.querySelector('.policy-box');
    if (!input) return;

    html2canvas(input as HTMLElement).then((canvas: HTMLCanvasElement) => {
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('privacy-policy.pdf');
    });
  };

  const [openSections, setOpenSections] = useState<{ [key: string]: boolean }>({});
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    if (stored === "dark") setIsDarkMode(true);
  }, []);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const toggleTheme = () => setIsDarkMode((prev) => !prev);

  const sections = [
    { key: "1", title: t("t1"), content: t("p1") },
    { key: "2", title: t("t2"), content: t("p2") },
    { key: "3", title: t("t3"), content: t("p3") },
    { key: "4", title: t("t4"), content: t("p4") },
    { key: "5", title: t("t5"), content: t("p5") },
    { key: "6", title: t("t6"), content: t("p6") },
  ];

  return (
    <div className="bg-gray-300 min-h-screen">
      <Header />
      <main className="font-montserrat policy-box">
        <div className="ml-[5%] mt-[5%]">
          <h1 className="font-bold text-title">{t("Privacy Policy")}</h1>
        </div>

        <div className="flex flex-col items-center gap-4 mt-6 mx-[5%]">
          <input
            type="text"
            placeholder="Search sections..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full max-w-md p-2 border border-gray-300 rounded"
          />

          <div className="flex gap-4 flex-wrap justify-center">
            <button onClick={expandAll} className="bg-green-600 text-white px-4 py-2 rounded">Expand All</button>
            <button onClick={collapseAll} className="bg-green-600 text-white px-4 py-2 rounded">Collapse All</button>
            <button onClick={downloadPDF} className="bg-green-600 text-white px-4 py-2 rounded">Download PDF</button>
          </div>
        </div>

        <div className="mx-[5%] mt-6">
          {filteredSections.length === 0 ? (
            <p>No matching sections found.</p>
          ) : (
            filteredSections.map((section, index) => (
              <AccordionItem
                key={index}
                title={t(section.titleKey)}
                content={t(section.contentKey)}
                isOpen={openIndexes.includes(index)}
                onClick={() => toggleAccordion(index)}
              />
            ))
          )}
        </div>

        <div className="flex items-center justify-center mt-24 mb-[-3rem]">
          <p className="text-center text-[14px] w-[80%]">{t("p7")}</p>
    <div className="flex flex-col min-h-screen bg-white text-gray-900 dark:bg-black dark:text-white transition-colors duration-300">
      <Header />

      <main className="flex-grow flex flex-col items-center font-montserrat relative pb-20">
        <h1 className="text-3xl font-bold mt-10 mb-6">{t("Privacy Policy")}</h1>

        <div className="w-full max-w-3xl px-4 rounded-lg p-6 bg-gray-200 text-gray-900 dark:bg-[#263238] dark:text-white">
          {sections.map(({ key, title, content }) => (
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
          ))}
        </div>

        <div className="flex items-center justify-center mt-10">
          <p className="text-center text-[14px] max-w-4xl">{t("p7")}</p>
        </div>

        <button
          onClick={toggleTheme}
          className="absolute bottom-5 right-5 p-3 bg-[#f0f0f0] rounded-full shadow-md hover:bg-[#e0e0e0] dark:bg-[#333333] dark:hover:bg-[#444444] transition"
          aria-label="Toggle Theme"
        >
          {isDarkMode ? (
            <HiSun className="text-yellow-400 text-xl" />
          ) : (
            <HiMoon className="text-gray-800 text-xl" />
          )}
        </button>
      </main>

      <Footer />
    </div>
  );
};

export default Privacypolicy;
