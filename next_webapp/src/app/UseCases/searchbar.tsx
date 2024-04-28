import { useState } from 'react';
import caseStudies from './database';

const SearchBar = ({ onSearch }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [category, setCategory] = useState('all'); // default category or whatever your logic is

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(searchTerm, category);
  };

  return (
    <div className="p-4 flex mb-4">
      <form onSubmit={handleSubmit} className="flex items-center w-full max-w-10xl space-x-2">
        <input
          type="search"
          placeholder="Case study name or category"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-l-lg rounded-r-lg focus:outline-none focus:border-green-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <label htmlFor="category-select" className="sr-only">All categories</label>
        <div className="flex-shrink-0">
        <select
          id="category-select"
          className="text-black border-2 border-gray-300 border-l-1 px-4 py-2 focus:outline-none focus:border-green-500 rounded-l-md rounded-r-lg"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        >
          <option value="all">All categories</option>
          {/* Add more options here based on your categories */}
          <option value="category1">Internet</option>
          <option value="category2">EV</option>
          <option value="category3">Security</option>
        </select>
        </div>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded-l-md rounded-r-lg hover:bg-green-600 focus:outline-none"
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default SearchBar;
