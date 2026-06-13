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
  const [activeSubtab, setActiveSubtab] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSupportOpen, setIsSupportOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  // Cargar metadatos y coordenadas al montar
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const statusRes = await fetch(`${API_URL}/api/status`);
        if (statusRes.ok) {
          setBackendStatus("ready");
        } else {
          setBackendStatus("offline");
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

  // Sincronizar sub-pestañas al cambiar de vista principal
  useEffect(() => {
    if (currentView === "predictor") {
      setActiveSubtab("Simulación");
    } else if (currentView === "explain") {
      setActiveSubtab("Global SHAP");
    } else {
      setActiveSubtab("");
    }
  }, [currentView]);

  // Manejar tema claro/oscuro
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.add("light");
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  // Navegar al predictor desde el mapa
  const handleSelectDepartment = (iso, dept) => {
    let countryName = iso;
    if (metadata) {
      const found = Object.entries(metadata).find(([name, data]) => data.iso_a0 === iso);
      if (found) {
        countryName = found[0];
      }
    }
    setSelectedCountry(countryName);
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
            darkMode={darkMode}
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
            activeSubtab={activeSubtab}
            backendStatus={backendStatus}
          />
        );
      case "explain":
        return <ExplainabilityView activeSubtab={activeSubtab} />;
      case "info":
        return <InfoView />;
      default:
        return (
          <DashboardView
            coordinates={coordinates}
            onSelectDepartment={handleSelectDepartment}
            backendUrl={API_URL}
            darkMode={darkMode}
          />
        );
    }
  };

  return (
    <>
      {/* Sidebar — fixed, hidden on mobile */}
      <Sidebar 
        currentView={currentView} 
        setCurrentView={setCurrentView} 
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenSupport={() => setIsSupportOpen(true)}
      />

      {/* Main Content Area */}
      <main className="md:pl-64 flex flex-col min-h-screen bg-background text-on-background">
        {/* TopNavBar */}
        <Topbar
          currentView={currentView}
          activeSubtab={activeSubtab}
          setActiveSubtab={setActiveSubtab}
          onOpenSettings={() => setIsSettingsOpen(true)}
          darkMode={darkMode}
          setDarkMode={setDarkMode}
        />

        {/* Backend Offline Banner */}
        {backendStatus === "offline" && (
          <div className="bg-error-container/50 border-b border-outline-variant px-lg py-2 flex items-center gap-sm">
            <span className="material-symbols-outlined text-on-error-container text-[18px]">cloud_off</span>
            <p className="text-[12px] text-on-error-container font-medium">
              Modo Demo Activo (Servidor Local Desconectado). Inicia el backend ejecutando{" "}
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
          <p>© {new Date().getFullYear()} EpiPredict Dengue — Unidad de Análisis SMA-ML/DL</p>
          <div className="flex gap-lg">
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Documentación</span>
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Glosario</span>
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Soporte</span>
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

      {/* MODAL DE AJUSTES */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-2xl p-lg w-full max-w-md shadow-2xl flex flex-col gap-md">
            <div className="flex justify-between items-center border-b border-outline-variant pb-xs">
              <h3 className="text-headline-md text-primary font-bold flex items-center gap-xs">
                <span className="material-symbols-outlined">settings</span> Ajustes del Sistema
              </h3>
              <button onClick={() => setIsSettingsOpen(false)} className="text-on-surface-variant hover:text-primary cursor-pointer">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            
            <div className="space-y-lg py-sm">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-bold text-on-background text-label-md">Modo Oscuro</p>
                  <p className="text-body-md text-on-surface-variant">Alterna el tema visual de la interfaz</p>
                </div>
                <button 
                  onClick={() => setDarkMode(!darkMode)}
                  className={`w-12 h-6 rounded-full p-0.5 transition-colors duration-200 cursor-pointer ${darkMode ? "bg-primary flex justify-end" : "bg-outline flex justify-start"}`}
                >
                  <span className="w-5 h-5 rounded-full bg-white shadow-md"></span>
                </button>
              </div>

              <div className="space-y-xs">
                <p className="font-bold text-on-background text-label-md">Servidor del API (Backend)</p>
                <div className="flex items-center justify-between p-sm bg-surface-container-high rounded-xl">
                  <code className="text-body-md font-mono">{API_URL}</code>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${backendStatus === "ready" ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"}`}>
                    {backendStatus === "ready" ? "Conectado" : "Desconectado"}
                  </span>
                </div>
              </div>
            </div>

            <button 
              onClick={() => setIsSettingsOpen(false)}
              className="w-full bg-primary text-on-primary font-bold py-3 rounded-xl hover:opacity-90 transition-opacity cursor-pointer"
            >
              Guardar Cambios
            </button>
          </div>
        </div>
      )}

      {/* MODAL DE SOPORTE */}
      {isSupportOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-2xl p-lg w-full max-w-lg shadow-2xl flex flex-col gap-md">
            <div className="flex justify-between items-center border-b border-outline-variant pb-xs">
              <h3 className="text-headline-md text-primary font-bold flex items-center gap-xs">
                <span className="material-symbols-outlined">help</span> Soporte & Glosario Técnico
              </h3>
              <button onClick={() => setIsSupportOpen(false)} className="text-on-surface-variant hover:text-primary cursor-pointer">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            
            <div className="space-y-md py-sm max-h-[450px] overflow-y-auto pr-xs">
              <div>
                <p className="font-bold text-primary text-label-md">¿Cómo funciona la Alerta Temprana?</p>
                <p className="text-body-md text-on-surface-variant mt-xs leading-relaxed">
                  El sistema es un predictor autorregresivo. Utiliza los casos reportados el mes anterior (Lag-1) y el historial climático para calcular la tasa de incidencia de dengue proyectada para el mes en curso. Esto otorga una ventana de 30 días a las autoridades sanitarias para prevenir brotes.
                </p>
              </div>
              <div className="border-t border-outline-variant pt-md">
                <p className="font-bold text-primary text-label-md">¿Qué significan las métricas?</p>
                <ul className="text-body-md text-on-surface-variant mt-xs space-y-sm list-disc pl-md">
                  <li><strong>Tasa de Incidencia:</strong> Número de casos registrados de dengue por cada 100,000 habitantes.</li>
                  <li><strong>R² (Coeficiente de Determinación):</strong> Porcentaje de variabilidad de los datos explicado por el modelo. A mayor R², mayor fidelidad predictiva (~75.41% para el Ensemble LightGBM + LSTM).</li>
                  <li><strong>MAE (Error Absoluto Medio):</strong> Promedio de las desviaciones absolutas entre la predicción y el caso real (~9.87 casos/100k hab. para el Ensemble).</li>
                </ul>
              </div>
              <div className="border-t border-outline-variant pt-md">
                <p className="font-bold text-primary text-label-md">Contacto del Investigador</p>
                <p className="text-body-md text-on-surface-variant mt-xs">
                  Desarrollado como Proyecto Final en la Facultad de Ingeniería de Sistemas e Informática (FISI) - UNMSM. Autor: <strong>Yeshua Chavez</strong> (yeshua.chavez@unmsm.edu.pe).
                </p>
              </div>
            </div>

            <button 
              onClick={() => setIsSupportOpen(false)}
              className="w-full bg-primary text-on-primary font-bold py-3 rounded-xl hover:opacity-90 transition-opacity cursor-pointer animate-pulse"
            >
              Cerrar Soporte
            </button>
          </div>
        </div>
      )}
    </>
  );
}
