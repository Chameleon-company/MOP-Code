'use client';

import React, { useState } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { fixLeafletIcon } from '@/library/fixLeafletIcon';
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
    <div style={{ fontFamily: 'Arial, sans-serif', backgroundColor: '#f4f6f8', padding: '20px' }}>
      {/* Page Header */}
      <h1 style={{ fontSize: '28px', marginBottom: '10px', color: '#666' }}>EV Infrastructure</h1>

      <div
        style={{
          background: "#f4f6f8",
          padding: "10px 20px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "20px",
        }}
      >
        <button
          onClick={() => (window.location.href = "/")}
          style={{
            background: "#28a745",
            color: "white",
            border: "none",
            borderRadius: "6px",
            padding: "8px 14px",
            fontSize: "16px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          {/* Back arrow icon */}
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
          </svg>
          Back
        </button>

        {/* Right: Search Bar */}
        <div style={{ display: "flex", alignItems: "center" }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search..."
            style={{
              padding: "6px 10px",
              borderRadius: "4px",
              border: "1px solid #999",
              marginRight: "10px",
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              padding: "6px 12px",
              background: "#007bff",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Search
          </button>
        </div>
      </div>

      {/* Incase if the search result is not found */}
      {noResults && (
        <p style={{ textAlign: "center", color: "red", marginBottom: "20px" }}>
          No search results found
        </p>
      )}

      <div
        style={{
          backgroundImage: "url('/img/ev-banner.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          padding: "140px",
          textAlign: "center",
          marginBottom: "20px",
          borderRadius:"25px",
          boxShadow: "20px",
        }}
      >
        <button style={buttonStyle}>Find Charging Station</button>
        <button style={{ ...buttonStyle, marginLeft: "20px" }}>View Live Data</button>
      </div>

      {/* Live Data Section */}
      <h3 style={{ textAlign: 'center', marginBottom: '10px' }}>VIEW LIVE DATA</h3>
        <div style={card}>
          <h4>Station Status Distribution</h4>
            <div style={{ marginTop: '10px' }}>
              <StatusBar label="Available" color="#4caf50" value={availablePercentage} />
              <StatusBar label="In Use" color="#ff9800" value={inUsePercentage} />
              <StatusBar label="Offline" color="#f44336" value={offlinePercentage} />
            </div>
              <h4 style={{ marginTop: '30px' }}>Recent Activity</h4>
            <div style={{ background: '#e0e0e0', height: '100px', marginTop: '10px' }} />
        </div>

      {/* Search & Filter */}
      <h3 style={{ textAlign: 'center', margin: '40px 0 10px' }}>FIND YOUR NEAREST STATION</h3>
      <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
        <input placeholder="Search" style={inputStyle} />
        <select style={inputStyle}>
          <option>Station Type</option>
        </select>
        <select style={inputStyle}>
          <option>Availability</option>
        </select>
        <button style={buttonStyle}>Filter</button>
      </div>
      {/* View Toggle */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '20px' }}>
        <button style={buttonStyle}>Map View</button>
        <button style={buttonStyle}>List View</button>
      </div>
      {/* Location Map */}
      <div style={{ height: '400px', marginBottom: '20px' }}>
        <MapContainer center={[-37.8136, 144.9631]} zoom={13} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {(filteredStations || allStations).map((station, i) => (
            <StationMarker key={`${station.name}-${i}`} station={station} index={i} />
          ))}
        </MapContainer>
      </div>
      <div style={{ textAlign: 'center' }}>
        <button style={buttonStyle}>Load More Stations</button>
      </div>
    </div>
  );
};

const StatusBar = ({label,color,value = 0,}: {label: string; color: string;value?: number;}) => (
  <div style={{ marginBottom: '10px' }}>
    <div
      style={{
        height: '10px',
        backgroundColor: color,
        width: `${Math.max(0, Math.min(100, value))}%`, // clamp 0..100
        transition: 'width 250ms ease',
      }}
    />
    <span style={{ fontSize: '14px' }}>{label}</span>
  </div>
);
const buttonStyle: React.CSSProperties = {
  padding: '10px 20px',
  backgroundColor: '#1976d2',
  color: 'white',
  border: 'none',
  borderRadius: '5px',
  cursor: 'pointer',
};

const inputStyle: React.CSSProperties = {
  padding: '10px',
  minWidth: '150px',
  borderRadius: '5px',
  border: '1px solid #ccc',
};

const card: React.CSSProperties = {
  backgroundColor: 'white',
  padding: '20px',
  maxWidth: '800px',
  margin: '0 auto 40px',
  borderRadius: '8px',
  boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
};

export default EVInfrastructurePage;