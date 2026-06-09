import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import DashboardView from "./components/DashboardView";
import PredictorView from "./components/PredictorView";
import ExplainabilityView from "./components/ExplainabilityView";
import InfoView from "./components/InfoView";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [currentView, setCurrentView] = useState("dashboard");
  const [metadata, setMetadata] = useState(null);
  const [coordinates, setCoordinates] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState("");
  const [selectedDept, setSelectedDept] = useState("");
  const [backendStatus, setBackendStatus] = useState("connecting");

  // Cargar metadatos y coordenadas al montar
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const statusRes = await fetch(`${API_URL}/api/status`);
        if (statusRes.ok) {
          setBackendStatus("ready");
        }

        const metaRes = await fetch(`${API_URL}/api/metadata`);
        if (metaRes.ok) {
          const metaData = await metaRes.json();
          setMetadata(metaData);
        }

        const coordRes = await fetch(`${API_URL}/api/coordinates`);
        if (coordRes.ok) {
          const coordData = await coordRes.json();
          setCoordinates(coordData);
        }
      } catch (err) {
        console.warn("Backend no disponible:", err.message);
        setBackendStatus("offline");
      }
    };

    loadInitialData();
  }, []);

  // Navegar al predictor desde el mapa
  const handleSelectDepartment = (iso, dept) => {
    setSelectedCountry(iso);
    setSelectedDept(dept);
    setCurrentView("predictor");
  };

  const renderView = () => {
    switch (currentView) {
      case "dashboard":
        return (
          <DashboardView
            coordinates={coordinates}
            onSelectDepartment={handleSelectDepartment}
            backendUrl={API_URL}
          />
        );
      case "predictor":
        return (
          <PredictorView
            metadata={metadata}
            selectedCountry={selectedCountry}
            selectedDept={selectedDept}
            setSelectedCountry={setSelectedCountry}
            setSelectedDept={setSelectedDept}
          />
        );
      case "explain":
        return <ExplainabilityView />;
      case "info":
        return <InfoView />;
      default:
        return (
          <DashboardView
            coordinates={coordinates}
            onSelectDepartment={handleSelectDepartment}
            backendUrl={API_URL}
          />
        );
    }
  };

  return (
    <>
      {/* Sidebar — fixed, hidden on mobile */}
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} />

      {/* Main Content Area — offset for sidebar on md+ */}
      <main className="md:pl-64 flex flex-col min-h-screen bg-background">
        {/* TopNavBar */}
        <Topbar currentView={currentView} />

        {/* Backend Offline Banner */}
        {backendStatus === "offline" && (
          <div className="bg-error-container/50 border-b border-outline-variant px-lg py-2 flex items-center gap-sm">
            <span className="material-symbols-outlined text-on-error-container text-[18px]">cloud_off</span>
            <p className="text-[12px] text-on-error-container font-medium">
              No se pudo conectar al backend ({API_URL}). Ejecuta{" "}
              <code className="bg-white/50 px-1 rounded text-[11px] font-mono">python -m uvicorn backend.main:app --reload</code>
            </p>
          </div>
        )}

        {/* Scrollable Content */}
        <section className="flex-1 overflow-y-auto px-lg pt-lg pb-xl">
          <div className="max-w-[1440px] mx-auto">
            {renderView()}
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-outline-variant px-lg py-md flex flex-col sm:flex-row justify-between items-center gap-md text-on-surface-variant text-label-md">
          <p>© 2024 Epidemiología Predictiva — Unidad de Análisis SMA-ML/DL</p>
          <div className="flex gap-lg">
            <span className="hover:text-primary transition-colors cursor-pointer">Documentación</span>
            <span className="hover:text-primary transition-colors cursor-pointer">Glosario</span>
            <span className="hover:text-primary transition-colors cursor-pointer">Privacidad</span>
          </div>
        </footer>
      </main>

      {/* Mobile Bottom Nav */}
      <nav className="fixed bottom-0 left-0 right-0 bg-surface border-t border-outline-variant md:hidden flex justify-around py-sm z-50">
        {[
          { id: "dashboard", icon: "dashboard", label: "Home" },
          { id: "predictor", icon: "query_stats", label: "Predicción" },
          { id: "explain", icon: "psychology", label: "XAI" },
          { id: "info", icon: "settings", label: "Más" },
        ].map((item) => (
          <div
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            className={`flex flex-col items-center gap-xs cursor-pointer ${
              currentView === item.id ? "text-primary font-bold" : "text-on-surface-variant"
            }`}
          >
            <span
              className="material-symbols-outlined"
              style={currentView === item.id ? { fontVariationSettings: "'FILL' 1" } : {}}
            >
              {item.icon}
            </span>
            <span className="text-[10px]">{item.label}</span>
          </div>
        ))}
      </nav>
    </>
  );
}
