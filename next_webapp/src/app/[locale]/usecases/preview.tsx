"use client";

import React, { useMemo, useState, useEffect } from "react";
import {
  ArrowLeftCircle,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  FileText,
  Sparkles,
} from "lucide-react";
import { CaseStudy } from "../../types";

const ITEMS_PER_PAGE = 9;

interface CardProps {
  study: CaseStudy;
  onClick: () => void;
}

const UseCaseCard: React.FC<CardProps> = ({ study, onClick }) => {
  const primaryTag = study.tags?.[0] || "Open Data";

  return (
    <div
      onClick={onClick}
      className="group cursor-pointer rounded-[24px] border border-gray-200 bg-white p-5 shadow-sm transition duration-300 hover:-translate-y-1 hover:border-green-400 hover:shadow-xl dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <span className="inline-flex rounded-full bg-green-50 px-3 py-1 text-xs font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
          {primaryTag}
        </span>

        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gray-100 text-green-600 dark:bg-gray-700 dark:text-green-300">
          <FileText size={20} />
        </div>
      </div>

      <h3 className="mb-3 text-xl font-bold leading-snug text-gray-900 dark:text-white">
        {study.name}
      </h3>

      <p className="mb-5 min-h-[72px] text-sm leading-6 text-gray-600 dark:text-gray-300">
        {study.description}
      </p>

      <div className="mb-5 flex flex-wrap gap-2">
        {study.tags?.slice(0, 3).map((tag) => (
          <span
            key={tag}
            className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between border-t border-gray-100 pt-4 dark:border-gray-700">
        <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          View overview
        </span>
        <ArrowRight className="text-green-600 transition group-hover:translate-x-1" size={18} />
      </div>
    </div>
  );
};

interface Props {
  caseStudies: CaseStudy[];
  trendingCaseStudies: CaseStudy[];
  selectedCaseStudy: CaseStudy | null;
  onSelectCaseStudy: (s: CaseStudy) => void;
  onBack: () => void;
}

const PreviewComponent: React.FC<Props> = ({
  caseStudies,
  selectedCaseStudy,
  onSelectCaseStudy,
  onBack,
}) => {
  const [page, setPage] = useState(1);

  useEffect(() => {
    setPage(1);
  }, [caseStudies]);

  const totalPages = Math.ceil(caseStudies.length / ITEMS_PER_PAGE);

  const visibleStudies = useMemo(() => {
    const start = (page - 1) * ITEMS_PER_PAGE;
    return caseStudies.slice(start, start + ITEMS_PER_PAGE);
  }, [caseStudies, page]);

  if (selectedCaseStudy) {
    return (
      <section className="rounded-[28px] border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-8">
        <button
          onClick={onBack}
          className="mb-6 inline-flex items-center gap-2 rounded-full bg-gray-100 px-4 py-2 text-sm font-semibold text-gray-700 transition hover:bg-gray-200 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
        >
          <ArrowLeftCircle size={18} />
          Back to all use cases
        </button>

        <div className="mb-6 inline-flex items-center rounded-full bg-green-50 px-4 py-1.5 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
          <Sparkles size={16} className="mr-2" />
          Selected Use Case
        </div>

        <h2 className="mb-4 text-3xl font-bold text-gray-900 dark:text-white">
          {selectedCaseStudy.name}
        </h2>

        <p className="mb-6 max-w-3xl text-base leading-7 text-gray-600 dark:text-gray-300">
          {selectedCaseStudy.description}
        </p>

        <div className="mb-8 flex flex-wrap gap-3">
          {selectedCaseStudy.tags?.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-gray-200 bg-gray-50 px-4 py-2 text-sm font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
            >
              {tag}
            </span>
          ))}
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl bg-[#f7f9fb] p-5 dark:bg-gray-700">
            <p className="mb-2 text-sm font-semibold text-gray-500 dark:text-gray-300">
              Focus
            </p>
            <p className="text-base font-semibold text-gray-900 dark:text-white">
              Practical open-data application
            </p>
          </div>

          <div className="rounded-2xl bg-[#f7f9fb] p-5 dark:bg-gray-700">
            <p className="mb-2 text-sm font-semibold text-gray-500 dark:text-gray-300">
              Format
            </p>
            <p className="text-base font-semibold text-gray-900 dark:text-white">
              Tiled case study preview
            </p>
          </div>

          <div className="rounded-2xl bg-[#f7f9fb] p-5 dark:bg-gray-700">
            <p className="mb-2 text-sm font-semibold text-gray-500 dark:text-gray-300">
              Keywords
            </p>
            <p className="text-base font-semibold text-gray-900 dark:text-white">
              {selectedCaseStudy.tags?.slice(0, 2).join(" • ") || "Open Data"}
            </p>
          </div>
        </div>
      </section>
    );
  }

  if (!caseStudies.length) {
    return (
      <section className="rounded-[28px] border border-dashed border-gray-300 bg-white px-6 py-16 text-center shadow-sm dark:border-gray-600 dark:bg-gray-800">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-green-50 text-green-600 dark:bg-green-900/30 dark:text-green-300">
          <FileText size={28} />
        </div>
        <h2 className="mb-3 text-2xl font-bold text-gray-900 dark:text-white">
          No use cases found
        </h2>
        <p className="mx-auto max-w-xl text-base leading-7 text-gray-600 dark:text-gray-300">
          Try another keyword or search mode. You can also reset the filters to
          explore all available use cases again.
        </p>
      </section>
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-[28px] border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-green-600 dark:text-green-300">
              Case Study Gallery
            </p>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Tiled showcase of use cases
            </h2>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              Showing {visibleStudies.length} of {caseStudies.length} use cases
            </p>
          </div>

          <div className="inline-flex items-center rounded-full bg-gray-100 px-4 py-2 text-sm font-semibold text-gray-700 dark:bg-gray-700 dark:text-white">
            Page {page} of {totalPages || 1}
          </div>
        </div>
      </section>

      <section className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
        {visibleStudies.map((study) => (
          <UseCaseCard
            key={study.id}
            study={study}
            onClick={() => onSelectCaseStudy(study)}
          />
        ))}
      </section>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => setPage((prev) => prev - 1)}
            disabled={page === 1}
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
          >
            <ChevronLeft size={18} />
          </button>

          <span className="rounded-full bg-green-50 px-4 py-2 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
            {page}
          </span>

          <button
            onClick={() => setPage((prev) => prev + 1)}
            disabled={page === totalPages}
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      )}
    </div>
  );
};

export default PreviewComponent;