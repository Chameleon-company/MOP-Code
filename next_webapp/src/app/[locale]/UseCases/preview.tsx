import React, { useState, useEffect } from "react";
import { CaseStudy } from "../../types";
import { ChevronLeft, ChevronRight, FileText, ArrowLeft } from "lucide-react";

const ITEMS_PER_PAGE = 9;

const PreviewComponent = ({ caseStudies }: { caseStudies: CaseStudy[] }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(undefined);

  const totalPages = Math.ceil(caseStudies.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const visibleCaseStudies = caseStudies.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
  };

  const handleCaseStudyClick = (study: CaseStudy) => {
    setSelectedCaseStudy(study);
  };

  const handleBack = () => {
    setSelectedCaseStudy(undefined);
  };

  if (selectedCaseStudy) {
    return (
      <div className="flex flex-col h-screen bg-gray-100 p-8 dark:bg-zinc-800">
        <button
          onClick={handleBack}
          className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
        >
          <ArrowLeft size={24} className="mr-2" />
          Back
        </button>
        <div className="bg-white rounded-lg shadow-md p-6 flex-grow overflow-hidden">
          <h1 className="text-3xl font-bold mb-4">{selectedCaseStudy.name}</h1>
          <iframe
            src={`/api?filename=${selectedCaseStudy.filename}`}
            className="w-full h-full border-none"
            title={selectedCaseStudy.name}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-100 p-8  dark:bg-zinc-800">
      <div className="grid grid-cols-3 gap-8 mb-8">
        {visibleCaseStudies.map((study) => (
          <div
            key={study.id}
            className="bg-white dark:bg-zinc-700 p-4 rounded-lg shadow-md cursor-pointer hover:shadow-lg transition-shadow duration-300"
            onClick={() => handleCaseStudyClick(study)}
          >
            <div className="flex items-center justify-center mb-4">
              <FileText size={48} className="text-green-500" />
              <FileText size={48} className="text-teal-400 -ml-6" />
              <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
            </div>
            <h3 className="font-bold text-lg text-center mb-2">{study.name}</h3>
            <p className="text-gray-600 text-center mb-2 dark:text-slate-300">{study.description}</p>
            <div className="flex flex-wrap justify-center gap-2">
            <p>Tags: </p>
              {study.tags.map((tag, index) => (
                <span
                  key={index}
                  className="bg-gray-200 text-gray-800 text-sm px-2 py-1 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-center items-center space-x-4">
        <button
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-2 bg-white  rounded-full shadow-md disabled:opacity-50"
        >
          <ChevronLeft size={24} className="dark:text-zinc-900"/>
        </button>
        <span className="text-xl font-semibold">{currentPage}</span>
        <button
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="p-2 bg-white rounded-full shadow-md disabled:opacity-50"
        >
          <ChevronRight size={24} className="dark:text-zinc-900" />
        </button>
      </div>
    </div>
  );
};

export default PreviewComponent;