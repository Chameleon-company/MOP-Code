import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const MELBOURNE_CENTER = [-37.8136, 144.9631];
const TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';

const LIGHT_COLORS = {
  pathway: '#3b82f6',     // blue
  intersection: '#f97316', // orange
  entry: '#22c55e',        // green
};

export default function DesignMap({ pathwayGeojson, lightPositions, parkBoundary, existingLights }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Initialize map only once
    if (!mapInstance.current) {
      mapInstance.current = L.map(mapRef.current, {
        center: MELBOURNE_CENTER,
        zoom: 15,
        scrollWheelZoom: true,
      });
      L.tileLayer(TILE_URL, { attribution: ATTRIBUTION, maxZoom: 19 }).addTo(mapInstance.current);
    }

    const map = mapInstance.current;
    const layers = [];

    // Park boundary
    if (parkBoundary && parkBoundary.coordinates) {
      const coords = parkBoundary.coordinates[0].map(c => [c[1], c[0]]);
      const boundary = L.polygon(coords, {
        color: '#22c55e',
        weight: 2,
        fillColor: '#22c55e',
        fillOpacity: 0.08,
        dashArray: '5,5',
      }).addTo(map);
      layers.push(boundary);
    }

    // Pathway polyline
    if (pathwayGeojson && pathwayGeojson.coordinates) {
      const pathCoords = pathwayGeojson.coordinates.map(c => [c[1], c[0]]);
      const pathway = L.polyline(pathCoords, {
        color: '#3b82f6',
        weight: 3,
        dashArray: '8,6',
        opacity: 0.8,
      }).addTo(map);
      layers.push(pathway);
    }

    // Existing streetlights (small grey dots)
    if (existingLights && existingLights.length > 0) {
      existingLights.forEach(light => {
        if (light.lat && light.lon) {
          const marker = L.circleMarker([light.lat, light.lon], {
            radius: 3,
            color: '#94a3b8',
            fillColor: '#94a3b8',
            fillOpacity: 0.5,
            weight: 1,
          }).bindPopup(`Existing light<br>Lux: ${light.lux || 'N/A'}`).addTo(map);
          layers.push(marker);
        }
      });
    }

    // Proposed light positions
    if (lightPositions && lightPositions.length > 0) {
      lightPositions.forEach(pos => {
        const color = LIGHT_COLORS[pos.type] || LIGHT_COLORS.pathway;

        // Coverage circle (translucent)
        const coverage = L.circle([pos.lat, pos.lng], {
          radius: 12,
          color: '#fbbf24',
          fillColor: '#fbbf24',
          fillOpacity: 0.12,
          weight: 0,
        }).addTo(map);
        layers.push(coverage);

        // Light marker
        const marker = L.circleMarker([pos.lat, pos.lng], {
          radius: 6,
          color: color,
          fillColor: color,
          fillOpacity: 0.9,
          weight: 2,
        }).bindPopup(
          `<strong>${pos.type.charAt(0).toUpperCase() + pos.type.slice(1)} Light</strong><br>` +
          `Chainage: ${pos.chainage_m}m`
        ).addTo(map);
        layers.push(marker);
      });
    }

    // Fit map to content bounds
    const allPoints = [];
    if (lightPositions) lightPositions.forEach(p => allPoints.push([p.lat, p.lng]));
    if (pathwayGeojson && pathwayGeojson.coordinates) {
      pathwayGeojson.coordinates.forEach(c => allPoints.push([c[1], c[0]]));
    }
    if (allPoints.length > 0) {
      map.fitBounds(L.latLngBounds(allPoints).pad(0.15));
    }

    // Cleanup layers on re-render
    return () => {
      layers.forEach(l => map.removeLayer(l));
    };
  }, [pathwayGeojson, lightPositions, parkBoundary, existingLights]);

  // Cleanup map on unmount
  useEffect(() => {
    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  // Don't render if no useful data
  if (!pathwayGeojson && (!lightPositions || lightPositions.length === 0)) {
    return null;
  }

  return (
    <div className="mt-3 rounded-lg overflow-hidden border border-slate-600">
      <div ref={mapRef} style={{ height: 400, width: '100%' }} />
      <div className="bg-slate-800 px-3 py-2 flex gap-4 text-xs text-slate-400">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" /> Pathway
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-orange-500 inline-block" /> Intersection
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" /> Entry Point
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-400 inline-block" /> Existing
        </span>
      </div>
    </div>
  );
}
