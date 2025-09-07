"use client"; // important for Next.js App Router

import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Custom marker icons (optional)
const parkingIcon = new L.Icon({
  iconUrl: "/icons/parking.png", // add icon image in /public/icons
  iconSize: [25, 25],
});
const safetyIcon = new L.Icon({
  iconUrl: "/icons/safety.png",
  iconSize: [25, 25],
});
const evIcon = new L.Icon({
  iconUrl: "/icons/ev.png",
  iconSize: [25, 25],
});

export default function CityMap() {
  return (
    <MapContainer
  center={[-37.8136, 144.9631]} // Melbourne CBD
  zoom={14}
  scrollWheelZoom={false}
  className="h-[400px] w-full rounded-xl shadow-md border border-gray-200"
>

      {/* Base tiles */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Example markers */}
      <Marker position={[-37.8136, 144.9631]} icon={parkingIcon}>
        <Popup>Parking Facility</Popup>
      </Marker>

      <Marker position={[-37.815, 144.97]} icon={safetyIcon}>
        <Popup>Safety Camera</Popup>
      </Marker>

      <Marker position={[-37.81, 144.96]} icon={evIcon}>
        <Popup>EV Charging Station</Popup>
      </Marker>
    </MapContainer>
  );
}
