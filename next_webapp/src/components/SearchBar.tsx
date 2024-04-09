"use client"

import React, { useState } from "react";
import "/public/styles/Searchbar.css";

const SearchBar = () => {
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearchInputChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleSearch = (e) => {
    e.preventDefault()
    // Redirect to the search results page with the current search value
    window.location.href = `/searchresults?q=${encodeURIComponent(
      searchQuery,
    )}`;
  };

  return (
    <form className="search-form" onSubmit={handleSearch}>
      <input
        type="text"
        className="search-input"
        placeholder="Search here..."
        value={searchQuery}
        onChange={handleSearchInputChange}
      />
      <button className="search-button" type="submit">
        <img src="/img/search.png" alt="Search Icon" onClick={handleSearch} />
      </button>
    </form>
  );
};

export default SearchBar;
