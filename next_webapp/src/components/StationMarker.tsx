'use client';

import React from 'react';
import { Marker, Popup } from 'react-leaflet';
import type { Station } from '../types/station';

type Props = {
  station: Station;
  index: number;
};

const StationMarker: React.FC<Props> = ({ station, index }) => (
  <Marker key={index} position={[station.lat, station.lng]}>
    <Popup minWidth={220}>
      <div>
        <div style={{ fontWeight: 600 }}>{station.name}</div>
        <div style={{ margin: '6px 0' }}>Status: {station.status}</div>
        <img
          src={station.image}
          alt={`${station.name} photo`}
          style={{ width: '100%', height: 'auto', borderRadius: 8, display: 'block' }}
        />
      </div>

        <a
            href={`https://www.google.com/maps?q=${station.lat},${station.lng}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: "0.85rem", color: "#1976d2" }}
          >
            Open in Google Maps
        </a>
    </Popup>
  </Marker>
);

export default StationMarker;