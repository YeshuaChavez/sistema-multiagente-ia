import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Corregir icono por defecto de Leaflet que a veces se rompe en Webpack/Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// Función para crear iconos interactivos y con badges de riesgo en Tailwind
const crearIconoRiesgo = (colorHex) => {
  return L.divIcon({
    html: `
      <div class="relative flex items-center justify-center w-8 h-8">
        <div class="absolute w-8 h-8 rounded-full opacity-40 animate-ping" style="background-color: ${colorHex};"></div>
        <div class="w-5 h-5 rounded-full border-2 border-white shadow-md" style="background-color: ${colorHex};"></div>
      </div>
    `,
    className: "custom-leaflet-marker",
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
};

// Componente para re-centrar el mapa dinámicamente con animación
function ChangeView({ coordinates }) {
  const map = useMap();
  useEffect(() => {
    if (!coordinates || coordinates.length === 0) return;
    
    // Si la lista de coordenadas tiene un país específico filtrado (menos de 80 marcadores)
    // y todos pertenecen al mismo código iso, centramos en el promedio de lat/lon y aplicamos zoom 5
    const firstIso = coordinates[0].iso_a0;
    const allSameCountry = coordinates.every(c => c.iso_a0 === firstIso);
    
    if (allSameCountry && coordinates.length < 80) {
      let latSum = 0;
      let lonSum = 0;
      let count = 0;
      coordinates.forEach(c => {
        if (c.lat && c.lon) {
          latSum += parseFloat(c.lat);
          lonSum += parseFloat(c.lon);
          count++;
        }
      });
      if (count > 0) {
        const avgLat = latSum / count;
        const avgLon = lonSum / count;
        map.setView([avgLat, avgLon], 5, { animate: true, duration: 1.2 });
      }
    } else {
      map.setView([-15.0, -65.0], 3, { animate: true, duration: 1.2 });
    }
  }, [coordinates, map]);
  return null;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Map({ coordinates, onSelectDepartment, backendUrl, darkMode }) {
  const [mapData, setMapData] = useState([]);

  const defaultPosition = [-15.0, -65.0];
  const defaultZoom = 3;

  useEffect(() => {
    if (!coordinates || coordinates.length === 0) return;

    const fetchRiskData = async () => {
      // Clave de lookup: "ISO_A0|ADM_NAME" → {color, nivel, mean_incidencia}
      let riskLookup = {};
      try {
        const res = await fetch(`${API_URL}/api/map-summary`);
        if (res.ok) {
          const summary = await res.json();
          for (const entry of summary) {
            const key = `${entry.iso_a0.trim().toUpperCase()}|${entry.adm_1_name.trim().toUpperCase()}`;
            riskLookup[key] = { color: entry.color, nivel: entry.nivel, mean_incidencia: entry.mean_incidencia };
          }
        }
      } catch (_) {}

      const updatedData = coordinates.map((coord) => {
        const key = `${String(coord.iso_a0 ?? "").trim().toUpperCase()}|${String(coord.adm_1_name ?? "").trim().toUpperCase()}`;
        const risk = riskLookup[key] ?? { color: "#10b981", nivel: "Normal", mean_incidencia: null };
        return { ...coord, ...risk };
      });
      setMapData(updatedData);
    };

    fetchRiskData();
  }, [coordinates]);

  return (
    <div className="w-full h-full relative">
      <MapContainer
        center={defaultPosition}
        zoom={defaultZoom}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <ChangeView coordinates={coordinates} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url={
            darkMode
              ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              : "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
          }
          key={darkMode ? "dark" : "light"}
        />

        {mapData.map((dept, idx) => {
          if (!dept.lat || !dept.lon) return null;
          return (
            <Marker
              key={`${dept.iso_a0}-${dept.adm_1_name}-${idx}`}
              position={[parseFloat(dept.lat), parseFloat(dept.lon)]}
              icon={crearIconoRiesgo(dept.color)}
            >
              <Popup>
                <div className="text-on-surface p-1 min-w-[160px]">
                  <h4 className="font-bold text-[14px] m-0">{dept.adm_1_name}</h4>
                  <p className="text-[11px] text-on-surface-variant m-0 uppercase font-semibold">
                    {dept.iso_a0}
                  </p>
                  <p className="text-[12px] my-1 font-bold" style={{ color: dept.color }}>
                    {dept.nivel}
                  </p>
                  {dept.mean_incidencia != null && (
                    <p className="text-[11px] text-on-surface-variant m-0">
                      Incidencia media: <strong>{dept.mean_incidencia} casos/100k</strong>
                    </p>
                  )}
                  <button
                    onClick={() => onSelectDepartment(dept.iso_a0, dept.adm_1_name)}
                    className="mt-2 w-full text-[11px] bg-primary text-white font-bold py-1 px-2 rounded hover:opacity-90 transition-colors cursor-pointer"
                  >
                    Abrir en Predictor
                  </button>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
