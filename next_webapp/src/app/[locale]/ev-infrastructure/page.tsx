'use client'
import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const EVInfrastructurePage = () => {
  // Dummy station data
  const stations = [
    { name: 'Federation Square EV Station', lat: -37.8136, lng: 144.9631, status: 'Available' },
    { name: 'Melbourne Central EV Station', lat: -37.815, lng: 144.965, status: 'In Use' },
  ];

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', backgroundColor: '#f4f6f8', padding: '20px' }}>
      {/* Page Header */}
      <h1 style={{ fontSize: '28px', marginBottom: '10px', color: '#666' }}>EV Infrastructure</h1>

      {/* Nav Bar Placeholder */}
      <div style={{ background: '#ccc', padding: '10px', textAlign: 'center', marginBottom: '20px' }}>
        Nav Bar
      </div>

      {/* Hero Section */}
      <div style={{ background: '#ddd', padding: '60px', textAlign: 'center', marginBottom: '20px' }}>
        <h2 style={{ marginBottom: '30px' }}>Photo</h2>
        <button style={buttonStyle}>Find Charging Station</button>
        <button style={{ ...buttonStyle, marginLeft: '20px' }}>View Live Data</button>
      </div>

      {/* Live Data Section */}
      <h3 style={{ textAlign: 'center', marginBottom: '10px' }}>VIEW LIVE DATA</h3>
      <div style={card}>
        <h4>Station Status Distribution</h4>
        <div style={{ marginTop: '10px' }}>
          <StatusBar label="Available" color="#4caf50" />
          <StatusBar label="In Use" color="#ff9800" />
          <StatusBar label="Offline" color="#f44336" />
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
          {stations.map((station, index) => (
            <Marker key={index} position={[station.lat, station.lng]}>
              <Popup>
                {station.name}

                Status: {station.status}
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {/* Load More */}
      <div style={{ textAlign: 'center' }}>
        <button style={buttonStyle}>Load More Stations</button>
      </div>
    </div>
  );
};

// Reusable status bar component
const StatusBar = ({ label, color }: { label: string; color: string }) => (
  <div style={{ marginBottom: '10px' }}>
    <div style={{ height: '10px', backgroundColor: color, width: '100%' }}></div>
    <span style={{ fontSize: '14px' }}>{label}</span>
  </div>
);

// Styles
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