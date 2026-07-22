// src/app/cafes-restaurants/page.tsx

'use client';
import { useEffect } from 'react';
import 'leaflet/dist/leaflet.css';
import './cafes-restaurants.css';

export default function CafesRestaurantsPage() {
  useEffect(() => {
    import('leaflet').then((L) => {
      setTimeout(() => {
        const map = L.map('map').setView([-37.8136, 144.9631], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors',
        }).addTo(map);

        // 加载数据
        fetch('/data/cafes_restaurants.json')
          .then((res) => res.json())
          .then((data) => {
            data.forEach((place: any) => {
              const color = place.type === 'cafe' ? 'blue' : 'orange';

              const icon = L.divIcon({
                className: 'custom-marker',
                html: `<div style="background:${color};width:14px;height:14px;border-radius:50%"></div>`,
              });

              L.marker([place.lat, place.lng], { icon })
                .addTo(map)
                .bindPopup(`<b>${place.name}</b><br/>${place.type}<br/>Rating: ${place.rating}`);
            });
          });

        // 图例
        const legend = (L as any).control({ position: 'bottomright' });
        legend.onAdd = function () {
          const div = L.DomUtil.create('div', 'cafes-legend');
          div.innerHTML = `
            <h4>Cafes & Restaurants</h4>
            <div class="legend-item">
              <span class="color-box" style="background-color: blue;"></span> Cafe
            </div>
            <div class="legend-item">
              <span class="color-box" style="background-color: orange;"></span> Restaurant
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
      <header className="page-title">Cafes & Restaurants</header>

      {/* Map Section */}
      <section className="section-gray-light">
        <h2>Cafes & Restaurants Map</h2>
        <p style={{ marginBottom: '1rem' }}>
          This map shows cafes and restaurants in Melbourne with their type and rating.
        </p>
        <div id="map" style={{ width: '100%', height: '400px', borderRadius: '10px' }}></div>
      </section>

      {/* Insights */}
      <section className="section-gray-light">
        <h2>Use Case Insights</h2>
        <p className="section-description">
          Explore trends of cafes and restaurants across Melbourne
        </p>

        <div className="insights-form">
          <select className="insights-select">
            <option value="">Select type</option>
            <option value="cafe">Cafe</option>
            <option value="restaurant">Restaurant</option>
          </select>

          <select className="insights-select">
            <option value="">All Melbourne</option>
            <option value="cbd">Melbourne CBD</option>
            <option value="docklands">Docklands</option>
            <option value="fitzroy">Fitzroy</option>
          </select>

          <button className="insights-button">Query</button>
        </div>

        <div className="insights-chart-placeholder">
          <p>Chart will be displayed here...</p>
        </div>
      </section>

      {/* Features */}
      <section className="section-gray-light">
        <h2>Features</h2>
        <div className="features-container">
          <div className="feature-card">
            <span className="feature-icon">⭐</span>
            Top Rated
          </div>
          <div className="feature-card">
            <span className="feature-icon">📍</span>
            Hotspots
          </div>
          <div className="feature-card">
            <span className="feature-icon">🕒</span>
            Opening Hours
          </div>
          <div className="feature-card">
            <span className="feature-icon">📊</span>
            Trends
          </div>
        </div>
      </section>

      {/* Upload */}
      <section className="section-gray-light">
        <div style={{ textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Upload and contribute
          </h3>
          <p style={{ fontSize: '1rem' }}>
            Share information about cafes & restaurants
          </p>
        </div>
      </section>
    </main>
  );
}
