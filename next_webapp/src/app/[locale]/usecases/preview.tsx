// app/usecases/preview.tsx
"use client";

import React, { useState } from "react";
import { ArrowLeftCircle, ArrowRightCircle, FileText } from "lucide-react";
import { CaseStudy } from "../../types";

const ITEMS_PER_PAGE = 6;
const CARD_W = "w-full sm:w-[260px]"; // full-width on xs, 260px on sm+
const CARD_H = "h-[180px]";

////////////////////////////////////////////////////////////////////////////////
// 1. Shared Card component
////////////////////////////////////////////////////////////////////////////////
interface CardProps {
  study: CaseStudy;
  onClick?: () => void;
}

const Card: React.FC<CardProps> = ({ study, onClick }) => (
  <div
    onClick={onClick}
    className={`${CARD_W} ${CARD_H} overflow-hidden bg-white dark:bg-dark border border-gray-200 dark:border-gray-600 shadow hover:shadow-lg transition-shadow flex flex-col justify-between p-4 cursor-pointer`}
  >
    {/* Icon */}
    <div className="flex justify-center mb-2">
      <FileText size={48} className="text-primary" />
      <FileText size={48} className="-ml-6 text-teal-400" />
      <FileText size={48} className="-ml-6 rotate-6 text-green-700" />
    </div>

    {/* Title */}
    <h3 className="text-sm font-semibold text-dark dark:text-white text-center whitespace-normal break-words">
      {study.name}
    </h3>

    {/* Description */}
    <p className="text-xs text-center text-gray-700 dark:text-gray-300 line-clamp-2">
      {study.description}
    </p>

    {/* Tags */}
    <div className="flex flex-wrap justify-center gap-1 mt-1">
      {study.tags.slice(0, 3).map((t) => (
        <span
          key={t}
          className="text-[10px] px-2 py-[1px] bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-white"
        >
          {t}
        </span>
      ))}
    </div>
  </div>
);

////////////////////////////////////////////////////////////////////////////////
// 2. Trending (hidden on <lg>, only first 2)
////////////////////////////////////////////////////////////////////////////////
interface TrendingProps {
  trending: CaseStudy[];
  onSelect: (s: CaseStudy) => void;
}

const Trending: React.FC<TrendingProps> = ({ trending, onSelect }) => (
  <div className="flex flex-col space-y-6">
    {trending.slice(0, 2).map((s) => (
      <Card key={s.id} study={s} onClick={() => onSelect(s)} />
    ))}
  </div>
);

////////////////////////////////////////////////////////////////////////////////
// 3. Main PreviewComponent
////////////////////////////////////////////////////////////////////////////////
interface Props {
  caseStudies: CaseStudy[];
  trendingCaseStudies: CaseStudy[];
  selectedCaseStudy: CaseStudy | null;
  onSelectCaseStudy: (s: CaseStudy) => void;
  onBack: () => void;
}

const PreviewComponent: React.FC<Props> = ({
  caseStudies,
  trendingCaseStudies,
  selectedCaseStudy,
  onSelectCaseStudy,
  onBack,
}) => {
  const [page, setPage] = useState(1);

  // Detail view
  if (selectedCaseStudy) {
    return (
      <div className="flex flex-col min-h-screen p-8 bg-white dark:bg-[#263238]">
        <button
          onClick={onBack}
          className="flex items-center text-primary mb-4 hover:text-primary/80"
        >
          <ArrowLeftCircle size={24} className="mr-2" /> Back
        </button>
        <div className="flex-grow overflow-auto bg-white dark:bg-dark shadow p-6">
          <h1 className="text-3xl font-bold mb-4 text-dark dark:text-white">
            {selectedCaseStudy.name}
          </h1>
          <iframe
            src={`/api?filename=${selectedCaseStudy.filename}`}
            title={selectedCaseStudy.name}
            className="w-full h-[70vh] border-none"
          />
        </div>
      </div>
    );
  }

  // No results
  if (!caseStudies.length) {
    return <p className="p-8 text-center text-lg">No use cases found.</p>;
  }

  // Grid view
  const totalPages = Math.ceil(caseStudies.length / ITEMS_PER_PAGE);
  const visible = caseStudies.slice(
    (page - 1) * ITEMS_PER_PAGE,
    page * ITEMS_PER_PAGE
  );

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col lg:flex-row gap-8">
        {/* Trending sidebar */}
        <aside className="hidden lg:block bg-gray-200 dark:bg-gray-700 p-6">
          <h2 className="mb-4 text-lg font-semibold text-dark dark:text-white">
            Trending
          </h2>
          <Trending
            trending={trendingCaseStudies}
            onSelect={onSelectCaseStudy}
          />
        </aside>

        {/* Main grid */}
        <div className="flex-1 bg-gray-200 dark:bg-gray-700 p-6">
          <div className="grid gap-y-6 gap-x-8 grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 auto-rows-[180px] justify-items-center">
            {visible.map((s) => (
              <Card key={s.id} study={s} onClick={() => onSelectCaseStudy(s)} />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex justify-center items-center space-x-4 mt-6">
              <button
                onClick={() => setPage(page - 1)}
                disabled={page === 1}
                className="disabled:opacity-50 p-1"
              >
                <ArrowLeftCircle
                  size={20}
                  className="text-dark dark:text-white"
                />
              </button>
              <span className="text-xl font-semibold text-dark dark:text-white">
                {page}
              </span>
              <button
                onClick={() => setPage(page + 1)}
                disabled={page === totalPages}
                className="disabled:opacity-50 p-1"
              >
                <ArrowRightCircle
                  size={20}
                  className="text-dark dark:text-white"
                />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviewComponent;
