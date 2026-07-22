import React, { useEffect, useRef, useState } from "react";
import { CATEGORY } from "../../types";
import { ChevronDown, Search } from "lucide-react";

export type LocalSearchMode = "title" | "tag" | "content";

interface SearchBarProps {
  onSearch: (term: string, mode: LocalSearchMode, category: CATEGORY) => void;
  initialTerm?: string;
  initialMode?: LocalSearchMode;
  onTermChange?: (term: string, mode: LocalSearchMode) => void;
}

const searchModeOptions: { value: LocalSearchMode; label: string }[] = [
  { value: "title", label: "Search by title" },
  { value: "tag", label: "Search by tag" },
  { value: "content", label: "Search by content" },
];

const SearchBar: React.FC<SearchBarProps> = ({ onSearch, initialTerm = "", initialMode = "title", onTermChange }) => {
  const [term, setTerm] = useState(initialTerm);
  const [mode, setMode] = useState<LocalSearchMode>(initialMode);
  const [category] = useState<CATEGORY>(CATEGORY.ALL);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const dropdownRef = useRef<HTMLDivElement | null>(null);

  const selectedMode =
    searchModeOptions.find((option) => option.value === mode) ??
    searchModeOptions[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTerm(e.target.value);
    onTermChange?.(e.target.value, mode);
  };

  const handleModeSelect = (newMode: LocalSearchMode) => {
    setMode(newMode);
    setIsDropdownOpen(false);
    onTermChange?.(term, newMode);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(term, mode, category);
  };

  const handleReset = () => {
    setTerm("");
    setMode("title");
    setIsDropdownOpen(false);
    onSearch("", "title", CATEGORY.ALL);
    onTermChange?.("", "title");
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
            onChange={handleInputChange}
            className="h-12 w-full rounded-2xl border border-gray-200 bg-gray-50 pl-12 pr-4 text-sm outline-none transition focus:border-green-500 focus:bg-white dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:focus:bg-gray-700"
          />
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div ref={dropdownRef} className="relative min-w-[210px]">
            <button
              type="button"
              onClick={() => setIsDropdownOpen((prev) => !prev)}
              className="flex h-12 w-full items-center justify-between rounded-2xl border border-gray-200 bg-white px-4 text-sm font-medium text-gray-800 shadow-sm outline-none transition hover:border-green-400 hover:bg-green-50 focus:border-green-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
              aria-haspopup="listbox"
              aria-expanded={isDropdownOpen}
            >
              <span>{selectedMode.label}</span>
              <ChevronDown
                size={18}
                className={`text-gray-500 transition-transform duration-200 ${
                  isDropdownOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {isDropdownOpen && (
              <div
                className="absolute left-0 top-full z-30 mt-2 w-full overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900"
                role="listbox"
              >
                <div className="h-0.5 w-full bg-gradient-to-r from-green-500 to-emerald-500" />

                <div className="py-1.5">
                  {searchModeOptions.map((option) => {
                    const isSelected = option.value === mode;

                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => handleModeSelect(option.value)}
                        className={`block w-full px-4 py-2.5 text-left text-sm font-medium transition ${
                          isSelected
                            ? "bg-gradient-to-r from-green-500 to-emerald-600 text-white"
                            : "text-gray-700 hover:bg-green-50 hover:text-green-700 dark:text-gray-200 dark:hover:bg-green-900/25 dark:hover:text-green-300"
                        }`}
                        role="option"
                        aria-selected={isSelected}
                      >
                        {option.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="h-12 rounded-2xl bg-gradient-to-r from-green-500 to-emerald-600 px-7 text-sm font-semibold text-white shadow-md shadow-green-500/20 transition hover:scale-[1.02] hover:shadow-lg active:scale-[0.98]"
          >
            Search
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="h-12 rounded-2xl border border-gray-200 bg-white px-7 text-sm font-semibold text-gray-700 shadow-sm transition hover:border-green-300 hover:bg-green-50 hover:text-green-700 active:scale-[0.98] dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  );
};

export default SearchBar;
