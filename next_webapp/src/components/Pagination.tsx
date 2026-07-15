"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationProps {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  total?: number;
  pageSize?: number;
  variant?: "public" | "admin";
  className?: string;
}

export default function Pagination({
  page,
  totalPages,
  onPageChange,
  total,
  pageSize,
  variant = "public",
  className = "",
}: PaginationProps) {
  if (totalPages <= 1) return null;

  if (variant === "admin") {
    const hasRange = total !== undefined && pageSize !== undefined;
    const rangeStart = hasRange ? (page - 1) * pageSize! + 1 : undefined;
    const rangeEnd = hasRange ? Math.min(page * pageSize!, total!) : undefined;

    return (
      <div className={`mt-6 flex items-center justify-between ${className}`}>
        {hasRange ? (
          <p className="text-sm text-[#687280]">
            Showing {rangeStart}–{rangeEnd} of {total}
          </p>
        ) : (
          <span />
        )}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page === 1}
            className="inline-flex items-center gap-1 rounded-lg border border-[#CFEFD9] bg-white px-3 py-2 text-sm text-[#1F8F50] transition hover:bg-[#DFF7E8] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ChevronLeft size={16} />
            Previous
          </button>
          <span className="text-sm text-[#687280]">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page === totalPages}
            className="inline-flex items-center gap-1 rounded-lg border border-[#CFEFD9] bg-white px-3 py-2 text-sm text-[#1F8F50] transition hover:bg-[#DFF7E8] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`mt-12 flex items-center justify-center gap-3 ${className}`}>
      <button
        onClick={() => onPageChange(page - 1)}
        disabled={page === 1}
        className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
      >
        <ChevronLeft size={18} />
      </button>
      <span className="rounded-full bg-green-50 px-4 py-2 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
        Page {page} of {totalPages}
      </span>
      <button
        onClick={() => onPageChange(page + 1)}
        disabled={page === totalPages}
        className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
      >
        <ChevronRight size={18} />
      </button>
    </div>
  );
}
