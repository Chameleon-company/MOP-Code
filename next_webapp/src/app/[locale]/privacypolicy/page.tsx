"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { storage } from "@/utils/storage";

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");

  const [searchTerm, setSearchTerm] = useState<string>("");
  const [openSections, setOpenSections] = useState<{ [key: string]: boolean }>({});
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const policyContentRef = useRef<HTMLDivElement>(null);

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  useEffect(() => {
    const stored = storage.getItem("theme");
    if (stored === "dark") setIsDarkMode(true);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (isDarkMode) {
      root.classList.add("dark");
      storage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      storage.setItem("theme", "light");
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

  const waitForLayout = () =>
    new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    );

  const downloadPDF = async () => {
    if (!policyContentRef.current || isGeneratingPDF) return;

    const previousSearchTerm = searchTerm;
    const previousOpenSections = openSections;

    try {
      setIsGeneratingPDF(true);
      setSearchTerm("");
      setOpenSections(
        Object.fromEntries(sections.map(({ key }) => [key, true]))
      );

      await waitForLayout();
      await document.fonts?.ready;

      const input = policyContentRef.current;
      if (!input) return;

      const canvas = await html2canvas(input, {
        backgroundColor: "#ffffff",
        scale: Math.min(window.devicePixelRatio || 1, 2),
        useCORS: true,
        logging: false,
        width: input.scrollWidth,
        height: input.scrollHeight,
        windowWidth: input.scrollWidth,
        windowHeight: input.scrollHeight,
        onclone: (clonedDocument) => {
          clonedDocument.documentElement.classList.remove("dark");
        },
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const renderedHeight = (canvas.height * pdfWidth) / canvas.width;
      let remainingHeight = renderedHeight;
      let yPosition = 0;

      pdf.addImage(imgData, "PNG", 0, yPosition, pdfWidth, renderedHeight);
      remainingHeight -= pageHeight;

      while (remainingHeight > 0) {
        yPosition = remainingHeight - renderedHeight;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 0, yPosition, pdfWidth, renderedHeight);
        remainingHeight -= pageHeight;
      }

      pdf.save("privacy-policy.pdf");
    } finally {
      setSearchTerm(previousSearchTerm);
      setOpenSections(previousOpenSections);
      setIsGeneratingPDF(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-white text-gray-900 dark:bg-black dark:text-white transition-colors duration-300">
      <Header />

      <main className="flex-grow flex flex-col items-center font-montserrat relative pb-20">
        <div
          ref={policyContentRef}
          id="privacy-policy-content"
          className="flex w-full flex-col items-center bg-white px-4 pb-4 text-gray-900"
        >
        <h1 className="text-3xl font-bold mt-10 mb-6">{t("Privacy Policy")}</h1>

        <div className="flex flex-col items-center gap-4 w-full max-w-4xl">
          <input
            data-html2canvas-ignore
            type="text"
            placeholder="Search sections..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded dark:bg-gray-800 dark:border-gray-600"
          />

          <div data-html2canvas-ignore className="flex gap-3 flex-wrap justify-center">
            <button onClick={expandAll} className="rounded-xl bg-green-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-green-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2">
              Expand All
            </button>
            <button onClick={collapseAll} className="rounded-xl bg-green-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-green-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2">
              Collapse All
            </button>
            <button
              onClick={downloadPDF}
              disabled={isGeneratingPDF}
              className="rounded-xl bg-green-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-green-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-60"
            >
              {isGeneratingPDF ? "Generating PDF..." : "Download PDF"}
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
                    {openSections[key] ? (
                      <ChevronUp className="h-5 w-5" />
                    ) : (
                      <ChevronDown className="h-5 w-5" />
                    )}
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
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Privacypolicy;
