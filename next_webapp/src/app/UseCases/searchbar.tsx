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
    <div className="p-4 flex">
      <form onSubmit={handleSubmit} className="flex items-center w-full">
        <input
          type="search"
          placeholder="Case study name or category"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-l-lg focus:outline-none focus:border-green-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <select
          className="text-black border-2 border-gray-300 border-l-0 px-4 py-2 focus:outline-none focus:border-green-500"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        >
          <option value="all">All categories</option>
          {/* Add more options here based on your categories */}
          <option value="category1">Internet</option>
          <option value="category2">EV</option>
          <option value="category3">Security</option>
        </select>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded-r-lg hover:bg-green-600 focus:outline-none"
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default SearchBar;
