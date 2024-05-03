import { useState } from "react";
import { CATEGORY, SEARCH_MODE } from "../types";

const SearchBar = ({
  onSearch,
}: {
  onSearch: (
    searchTerm: string,
    searchMode: SEARCH_MODE,
    category: CATEGORY
  ) => void;
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [category, setCategory] = useState(CATEGORY.ALL);
  const [searchMode, setSearchMode] = useState(SEARCH_MODE.TITLE);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(searchTerm, searchMode, category);
  };

  return (
    <div className="py-4 flex">
      <form onSubmit={handleSubmit} className="flex items-center w-full">
        <input
          type="search"
          placeholder="Enter use case name"
          className="w-full px-4 py-2 mr-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <select
          className="text-black mr-3 border-2 border-gray-300 px-4 py-2 rounded-lg focus:outline-none focus:border-green-500"
          value={category}
          onChange={(e) => setSearchMode(e.target.value as SEARCH_MODE)}
        >
          <option value={SEARCH_MODE.TITLE}>Search by title</option>
          <option value={SEARCH_MODE.CONTENT}>Search by content</option>
        </select>
        <select
          className="text-black mr-3 border-2 border-gray-300 px-4 py-2 rounded-lg focus:outline-none focus:border-green-500"
          value={category}
          onChange={(e) => setCategory(e.target.value as CATEGORY)}
        >
          <option value={CATEGORY.ALL}>All categories</option>
          {/* Add more options here based on your categories */}
          <option value={CATEGORY.INTERNET}>Internet</option>
          <option value={CATEGORY.EV}>EV</option>
          <option value={CATEGORY.SECURITY}>Security</option>
        </select>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 focus:outline-none"
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default SearchBar;
