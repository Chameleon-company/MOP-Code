'use client';
import { useEffect } from 'react';
import 'leaflet/dist/leaflet.css';
import './parking.css';

export default function ParkingPage() {
  useEffect(() => {
    import('leaflet').then((L) => {
      setTimeout(() => {
        const map = L.map('map').setView([-37.8136, 144.9631], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors',
        }).addTo(map);

        const defaultIcon = L.icon({
          iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
          iconSize: [25, 41],
          iconAnchor: [12, 41],
          popupAnchor: [1, -34],
          shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png',
          shadowSize: [41, 41],
        });

        L.marker([-37.8136, 144.9631], { icon: defaultIcon })
          .addTo(map)
          .bindPopup('Melbourne Central Parking Spot');

        const legend = (L as any).control({ position: 'bottomright' });
        legend.onAdd = function () {
          const div = L.DomUtil.create('div', 'safety-legend');
          div.innerHTML = `
            <h4>Parking Types</h4>
            <div class="legend-item">
              <span class="color-box" style="background-color: green;"></span> Available
            </div>
            <div class="legend-item">
              <span class="color-box" style="background-color: gray;"></span> Occupied
            </div>
            <div class="legend-item">
              <span class="color-box" style="background-color: blue;"></span> Disabled Zone
            </div>
          `;
          return div;
        };
        legend.addTo(map);
      }, 0);
    });
  }, []);

  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <header className="page-title">Parking</header>

      {/* Map View */}
      <section className="section-gray-light">
        <h2>Parking Map</h2>
        <p>This map shows available parking and facility types.</p>
        <div id="map" style={{ width: '100%', height: '400px', borderRadius: '10px' }}></div>
      </section>

      {/* Query */}
      <section className="section-gray-medium">
        <h2>Parking Query</h2>
        <p className="section-description">Select a region to view detailed parking data.</p>

        <div className="insights-form">
          <select className="insights-select">
            <option value="">Select Type</option>
            <option value="offstreet">Off-Street Parking</option>
            <option value="disabled">Disabled Access</option>
            <option value="motorbike">Motorbike Bays</option>
          </select>
          <select className="insights-select">
            <option value="">All Melbourne</option>
            <option value="cbd">Melbourne CBD</option>
            <option value="north">North Melbourne</option>
            <option value="south">South Melbourne</option>
          </select>
          <button className="insights-button">Search</button>
        </div>

        <div className="insights-chart-placeholder">
          <p>Chart output or location list will be shown here...</p>
        </div>
      </section>

      {/* Features */}
      <section className="section-gray-dark">
        <h2>Parking Features</h2>
        <div className="features-container">
          <div className="feature-card">
            <span className="feature-icon">🅿️</span>
            Parking Types
          </div>
          <div className="feature-card">
            <span className="feature-icon">🛵</span>
            Motorbike Zones
          </div>
          <div className="feature-card">
            <span className="feature-icon">♿</span>
            Disabled Access
          </div>
          <div className="feature-card">
            <span className="feature-icon">📊</span>
            Fine Statistics
          </div>
        </div>
      </section>

      {/* Upload */}
      <section className="section-gray-light">
        <div style={{ textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Upload and Contribute
          </h3>
          <p style={{ fontSize: '1rem' }}>
            Got parking data? Help improve access for all.
          </p>
        </div>
      </section>
    </main>
  );
}
