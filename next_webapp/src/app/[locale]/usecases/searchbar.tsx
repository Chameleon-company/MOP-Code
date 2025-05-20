// app/usecases/searchbar.tsx

import React, { useState } from "react";
import { CATEGORY, SEARCH_MODE } from "../../types";
import { useTranslations } from "next-intl";
import { Search } from "lucide-react";

interface SearchBarProps {
  onSearch: (term: string, mode: SEARCH_MODE, category: CATEGORY) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const [term, setTerm] = useState("");
  // empty initial value so placeholder shows
  const [mode, setMode] = useState<string>("");
  const [category, setCategory] = useState<CATEGORY>(CATEGORY.ALL);
  const t = useTranslations("usecases");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // if no mode selected, default to title
    const selectedMode = (mode as SEARCH_MODE) || SEARCH_MODE.TITLE;
    onSearch(term, selectedMode, category);
  };

  return (
    <div className="bg-gray-100 dark:bg-gray-700 py-2 px-4 mb-8 rounded">
      <form
        onSubmit={handleSubmit}
        className="flex flex-col sm:flex-row sm:items-center sm:space-x-3"
      >
        {/* SEARCH INPUT WITH ICON */}
        <div className="relative flex-1">
          <Search
            size={20}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-700 dark:text-gray-300"
          />
          <input
            type="search"
            placeholder={t("Case study name or category")}
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            className="
              w-full h-12
              pl-10 pr-4
              bg-transparent dark:bg-transparent
              focus:outline-none focus:border-primary
            "
          />
        </div>

        {/* SELECT + BUTTON */}
        <div className="mt-2 sm:mt-0 flex flex-col sm:flex-row sm:items-center sm:space-x-3">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="
              h-12 px-4
              bg-white dark:bg-gray-800
              border border-gray-300 dark:border-gray-600
              shadow
              text-black dark:text-white
              focus:outline-none focus:border-primary
            "
          >
            {/* placeholder option */}
            <option value="" disabled>
              Search Category
            </option>
            <option value={SEARCH_MODE.TITLE}>Search by title</option>
            <option value={SEARCH_MODE.CONTENT}>Search by tag</option>
          </select>

          <button
            type="submit"
            className="
              mt-2 sm:mt-0
              h-12 px-4
              bg-primary text-white
              hover:bg-primary/90
              focus:outline-none
              rounded
            "
          >
            {t("Search")}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SearchBar;
