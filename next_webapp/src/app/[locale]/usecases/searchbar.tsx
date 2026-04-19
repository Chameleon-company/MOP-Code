import React, { useState } from "react";
import { CATEGORY } from "../../types";
import { Search } from "lucide-react";

export type LocalSearchMode = "title" | "tag" | "content";

interface SearchBarProps {
  onSearch: (term: string, mode: LocalSearchMode, category: CATEGORY) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const [term, setTerm] = useState("");
  const [mode, setMode] = useState<LocalSearchMode>("title");
  const [category] = useState<CATEGORY>(CATEGORY.ALL);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(term, mode, category);
  };

  const handleReset = () => {
    setTerm("");
    setMode("title");
    onSearch("", "title", CATEGORY.ALL);
  };

  return (
    <div className="mb-8 rounded-[24px] border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-5">
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-4 lg:flex-row lg:items-center"
      >
        <div className="relative flex-1">
          <Search
            size={20}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400"
          />
          <input
            type="search"
            placeholder="Search use cases"
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            className="h-14 w-full rounded-2xl border border-gray-200 bg-gray-50 pl-12 pr-4 text-sm outline-none transition focus:border-green-500 focus:bg-white dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:focus:bg-gray-700"
          />
        </div>

        <div className="flex flex-col gap-3 sm:flex-row">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as LocalSearchMode)}
            className="h-14 min-w-[210px] rounded-2xl border border-gray-200 bg-white px-4 text-sm outline-none transition focus:border-green-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          >
            <option value="title">Search by title</option>
            <option value="tag">Search by tag</option>
            <option value="content">Search by content</option>
          </select>

          <button
            type="submit"
            className="h-14 rounded-2xl bg-primary px-6 text-sm font-semibold text-white transition hover:bg-primary/90"
          >
            Search
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="h-14 rounded-2xl border border-gray-200 bg-white px-6 text-sm font-semibold text-gray-700 transition hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  );
};

export default SearchBar;