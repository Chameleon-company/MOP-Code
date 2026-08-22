'use client';

import React, { useState } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { fixLeafletIcon } from '@/library/fixLeafletIcon';
import { Link } from '@/i18n-navigation';
fixLeafletIcon();

import { stations as allStations } from '../../../data/stations';
import StationMarker from '../../../components/StationMarker';
import type { Station } from '../../../types/station';

const EVInfrastructurePage = () => {
  const [query, setQuery] = useState('');
  const [filteredStations, setFilteredStations] = useState<Station[]>(allStations);
  const [noResults, setNoResults] = useState(false);

  const handleSearch = () => {
    const q = query.toLowerCase().trim();
    const results = q ? allStations.filter(s => s.name.toLowerCase().includes(q)) : allStations;
    setFilteredStations(results);
    setNoResults(results.length === 0);
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSearch();
    }
  };

  const total = filteredStations.length || 1;
  const availablePercentage = (filteredStations.filter(s => s.status.toLowerCase() === 'available').length / total) * 100;
  const inUsePercentage = (filteredStations.filter(s => s.status.toLowerCase() === 'in use').length / total) * 100;
  const offlinePercentage = (filteredStations.filter(s => s.status.toLowerCase() === 'offline').length / total) * 100;

  return (
    <div className="font-sans bg-gray-100 min-h-screen p-5 dark:bg-gray-900 text-gray-800 dark:text-gray-100">
      {/* Page Header */}
      <h1 className="text-2xl font-semibold mb-2.5 text-gray-600 dark:text-gray-300">EV Infrastructure</h1>

      <div className="bg-gray-100 dark:bg-gray-900 py-2.5 px-5 flex items-center justify-between mb-5">
        <Link
          href="/"
          className="bg-green-600 hover:bg-green-700 text-white rounded-md px-3.5 py-2 text-base font-medium flex items-center gap-2 cursor-pointer transition-colors"
        >
          {/* Back arrow icon */}
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
          </svg>
          Back
        </Link>

        {/* Right: Search Bar */}
        <div className="flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search..."
            className="px-2.5 py-1.5 rounded border border-gray-400 mr-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
          />
          <button
            onClick={handleSearch}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded cursor-pointer transition-colors text-sm font-medium"
          >
            Search
          </button>
        </div>
      </div>

      {/* Incase if the search result is not found */}
      {noResults && (
        <p className="text-center text-red-600 font-medium mb-5">
          No search results found
        </p>
      )}

      <div className="bg-cover bg-center bg-no-repeat py-32 text-center mb-5 rounded-3xl shadow-xl bg-[url('/img/ev-banner.png')]">
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors">
          Find Charging Station
        </button>
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors ml-5">
          View Live Data
        </button>
      </div>

      {/* Live Data Section */}
      <h3 className="text-center font-bold mb-2.5 text-base tracking-wider uppercase">VIEW LIVE DATA</h3>
      <div className="bg-white dark:bg-gray-800 p-5 max-w-3xl mx-auto mb-10 rounded-lg shadow-md">
        <h4 className="font-semibold text-base mb-2">Station Status Distribution</h4>
        <div className="mt-2.5">
          <StatusBar label="Available" color="#4caf50" value={availablePercentage} />
          <StatusBar label="In Use" color="#ff9800" value={inUsePercentage} />
          <StatusBar label="Offline" color="#f44336" value={offlinePercentage} />
        </div>
        <h4 className="font-semibold text-base mt-7">Recent Activity</h4>
        <div className="bg-gray-200 dark:bg-gray-700 h-25 mt-2.5 rounded-md" />
      </div>

      {/* Search & Filter */}
      <h3 className="text-center font-bold my-10 mb-2.5 text-base tracking-wider uppercase">FIND YOUR NEAREST STATION</h3>
      <div className="flex justify-center flex-wrap gap-2.5 mb-5">
        <input placeholder="Search" className="p-2.5 min-w-[150px] rounded-md border border-gray-300 dark:bg-gray-800 dark:border-gray-600 text-sm outline-none" />
        <select className="p-2.5 min-w-[150px] rounded-md border border-gray-300 dark:bg-gray-800 dark:border-gray-600 text-sm outline-none">
          <option>Station Type</option>
        </select>
        <select className="p-2.5 min-w-[150px] rounded-md border border-gray-300 dark:bg-gray-800 dark:border-gray-600 text-sm outline-none">
          <option>Availability</option>
        </select>
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors">
          Filter
        </button>
      </div>

      {/* View Toggle */}
      <div className="flex justify-center gap-2.5 mb-5">
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors">
          Map View
        </button>
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors">
          List View
        </button>
      </div>

      {/* Location Map */}
      <div className="h-[400px] mb-5 rounded-xl overflow-hidden shadow-sm">
        <MapContainer center={[-37.8136, 144.9631]} zoom={13} className="h-full w-full">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {(filteredStations || allStations).map((station, i) => (
            <StationMarker key={`${station.name}-${i}`} station={station} index={i} />
          ))}
        </MapContainer>
      </div>

      <div className="text-center">
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md cursor-pointer font-medium transition-colors">
          Load More Stations
        </button>
      </div>
    </div>
  );
};

const StatusBar = ({ label, color, value = 0 }: { label: string; color: string; value?: number }) => (
  <div className="mb-2.5">
    <div
      className="h-2.5 rounded-full transition-all duration-300 ease-out"
      style={{
        backgroundColor: color,
        width: `${Math.max(0, Math.min(100, value))}%`,
      }}
    />
    <span className="text-sm font-medium">{label}</span>
  </div>
);

export default EVInfrastructurePage;