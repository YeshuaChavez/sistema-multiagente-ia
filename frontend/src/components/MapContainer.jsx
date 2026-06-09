import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
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

export default function Map({ coordinates, onSelectDepartment, backendUrl }) {
  const [mapData, setMapData] = useState([]);

  // Centrar mapa en el centro geográfico aproximado de Latinoamérica
  const defaultPosition = [-15.0, -65.0];
  const defaultZoom = 3;

  useEffect(() => {
    if (!coordinates || coordinates.length === 0) return;

    // Para cada departamento, cargar su nivel de riesgo actual de forma asíncrona
    // o simular un nivel de riesgo basado en sus coordenadas si el endpoint falla,
    // pero idealmente llamamos a la API para cada depto
    const fetchIncidences = async () => {
      const updatedData = [];
      
      // Para optimizar en el prototipo, cargamos las coordenadas y les asignamos un riesgo aleatorio inicial
      // o el riesgo basado en sus datos reales si consultamos
      for (let coord of coordinates) {
        // Por defecto, color verde para riesgo normal
        let color = "#10b981"; 
        let nivel = "Bajo / Normal";
        
        // Simular algunos focos de calor realistas para la visualización del mapa
        const hash = coord.adm_1_name.length + coord.lat + coord.lon;
        if (hash % 7 === 0) {
          color = "#ef4444"; // Epidemia (rojo)
          nivel = "Epidemia";
        } else if (hash % 5 === 0) {
          color = "#f97316"; // Alerta (naranja)
          nivel = "Alerta";
        } else if (hash % 3 === 0) {
          color = "#eab308"; // Vigilancia (amarillo)
          nivel = "Vigilancia";
        }

        updatedData.push({
          ...coord,
          color,
          nivel
        });
      }
      setMapData(updatedData);
    };

    fetchIncidences();
  }, [coordinates]);

  return (
    <div className="w-full h-full relative">
      <MapContainer
        center={defaultPosition}
        zoom={defaultZoom}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
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
                <div className="text-on-surface p-1">
                  <h4 className="font-bold text-[14px] m-0">{dept.adm_1_name}</h4>
                  <p className="text-[11px] text-on-surface-variant m-0 uppercase font-semibold">
                    País: {dept.iso_a0}
                  </p>
                  <p className="text-[12px] my-1" style={{ color: dept.color }}>
                    <strong>Riesgo: {dept.nivel}</strong>
                  </p>
                  <button
                    onClick={() => onSelectDepartment(dept.iso_a0, dept.adm_1_name)}
                    className="mt-2 w-full text-[11px] bg-primary-container text-on-primary font-bold py-1 px-2 rounded hover:bg-primary transition-colors cursor-pointer"
                  >
                    Simular Predictor 🔮
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
