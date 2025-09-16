// src/app/safety/page.tsx

'use client';
import { useEffect } from 'react';
import 'leaflet/dist/leaflet.css';
import './safety.css';

export default function SafetyPage() {
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
        .bindPopup('Melbourne')
        .openPopup();

    const legend = (L as any).control({ position: 'bottomright' });
      legend.onAdd = function () {
     const div = L.DomUtil.create('div', 'safety-legend');
        div.innerHTML = `
            <h4>Safety Level</h4>
            <div class="legend-item">
            <span class="color-box" style="background-color: red;"></span> High Risk
            </div>
            <div class="legend-item">
            <span class="color-box" style="background-color: orange;"></span> Moderate Risk
            </div>
            <div class="legend-item">
            <span class="color-box" style="background-color: green;"></span> Safe Zone
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
    
    <header className="page-title">Safety</header>


      {/* map */}
      <section className="section-gray-light">
        <h2>Safety Map</h2>
        <p style={{ marginBottom: '1rem' }}> The map will indicate which areas are more or less safe based on their safety levels. </p>
        <div id="map" style={{ width: '100%', height: '400px', borderRadius: '10px' }}></div>
      </section>


    {/* Use Case Insights */}
    <section className="section-gray-light">
    <h2>Use Case Insights</h2>
    <p className="section-description">
        Explore safety-related data trends across Melbourne
    </p>


    <div className="insights-form">
        <select className="insights-select">
        <option value="">Select a category</option>
        <option value="crime">Crime Incidents</option>
        <option value="fire">Fire Reports</option>
        <option value="hospital">Hospital Access</option>
        {/* Choose queryable content based on available data. */}
        </select>

        {/* area */}
        <select className="insights-select">
        <option value="">All Melbourne</option>
        <option value="cbd">Melbourne CBD</option>
        <option value="north">North Melbourne</option>
        <option value="south">South Melbourne</option>
        <option value="east">East Melbourne</option>
        <option value="west">West Melbourne</option>
        {/* ... */}
        </select>

        <button className="insights-button">Query</button>
    </div>

    {/* Query Result Area */}
    <div className="insights-chart-placeholder">
        <p>Chart will be displayed here...</p>
    </div>
    </section>



    {/* Features */}
    <section className="section-gray-light">
    <h2>Features</h2>
    <div className="features-container">
        <div className="feature-card">
        <span className="feature-icon">🤖</span>
        Chatbot
        </div>
        <div className="feature-card">
        <span className="feature-icon">🚑</span>
        Emergency Access
        </div>
        <div className="feature-card">
        <span className="feature-icon">🌳</span>
        Urban Greening
        </div>
        <div className="feature-card">
        <span className="feature-icon">📶</span>
        Connectivity
        </div>
    </div>
    </section>

      
    {/* Upload  */}
    <section className="section-gray-light">
    <div style={{ textAlign: 'center' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
          Upload and contribute
        </h3>
        <p style={{ fontSize: '1rem' }}>
         Information you want to share
     </p>
    </div>
    </section>

    </main>
  );
}
