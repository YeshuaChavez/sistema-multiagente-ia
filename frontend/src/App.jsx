import React, { useState, useEffect } from "react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import BottomNav from "./components/BottomNav";
import DashboardView from "./components/DashboardView";
import PredictorView from "./components/PredictorView";
import ExplainabilityView from "./components/ExplainabilityView";
import InfoView from "./components/InfoView";
import MosquitoIcon from "./components/MosquitoIcon";

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
  const [simulationHistory, setSimulationHistory] = useState([]);
  const [isAppLoading, setIsAppLoading] = useState(true);

  // Cargar metadatos y coordenadas al montar
  useEffect(() => {
    const startTime = Date.now();
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
      } finally {
        const elapsedTime = Date.now() - startTime;
        const minDuration = 2200; // 2.2 segundos para mostrar la animación
        const remainingTime = Math.max(0, minDuration - elapsedTime);
        setTimeout(() => {
          setIsAppLoading(false);
        }, remainingTime);
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

  // Generar reporte consolidado PDF desde la sidebar
  const handleGenerateReport = async () => {
    const doc = new jsPDF();
    const fecha = new Date().toLocaleDateString("es-ES", { year: "numeric", month: "long", day: "numeric" });
    const PRIMARY = [30, 58, 95];
    const ORANGE  = [217, 119, 6];
    const GRAY    = [100, 100, 100];

    // Header
    doc.setFontSize(20);
    doc.setTextColor(...PRIMARY);
    doc.text("DenguePredict", 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(...GRAY);
    doc.text("Reporte Consolidado — Sistema Multi-Agente SMA-ML/DL", 14, 25);
    doc.text(`Generado: ${fecha}`, 14, 31);

    // Métricas del sistema
    let metrics = {
      records: "16,224", paises: "8", departamentos: "169",
      r2_xgb: "91.49%", r2_lstm: "90.35%", r2_ens: "91.47%",
      mae: "5.83", rmse: "20.67",
      mae_xgb: "6.07", rmse_xgb: "22.18",
      mae_lstm: "6.02", rmse_lstm: "20.52",
    };
    try {
      const r = await fetch(`${API_URL}/api/metrics`);
      if (r.ok) {
        const m = await r.json();
        metrics = {
          records:       m.records_procesados?.toLocaleString()                        ?? metrics.records,
          paises:        m.n_paises?.toString()                                         ?? metrics.paises,
          departamentos: m.n_departamentos?.toString()                                  ?? metrics.departamentos,
          r2_xgb:        m.r2_xgb      != null ? `${(m.r2_xgb      * 100).toFixed(2)}%` : metrics.r2_xgb,
          r2_lstm:       m.r2_lstm     != null ? `${(m.r2_lstm     * 100).toFixed(2)}%` : metrics.r2_lstm,
          r2_ens:        m.r2_ensemble != null ? `${(m.r2_ensemble * 100).toFixed(2)}%` : metrics.r2_ens,
          mae:           m.mae_ensemble  != null ? m.mae_ensemble.toFixed(2)             : metrics.mae,
          rmse:          m.rmse_ensemble != null ? m.rmse_ensemble.toFixed(2)            : metrics.rmse,
          mae_xgb:       m.mae_xgb      != null ? m.mae_xgb.toFixed(2)                  : metrics.mae_xgb,
          rmse_xgb:      m.rmse_xgb     != null ? m.rmse_xgb.toFixed(2)                 : metrics.rmse_xgb,
          mae_lstm:      m.mae_lstm     != null ? m.mae_lstm.toFixed(2)                  : metrics.mae_lstm,
          rmse_lstm:     m.rmse_lstm    != null ? m.rmse_lstm.toFixed(2)                 : metrics.rmse_lstm,
        };
      }
    } catch (_) {}

    doc.setFontSize(12);
    doc.setTextColor(...PRIMARY);
    doc.text("1. Métricas del Sistema", 14, 40);
    autoTable(doc, {
      startY: 44,
      head: [["Indicador", "Valor"]],
      body: [
        ["Registros históricos procesados", `${metrics.records} obs. (2014–2022)`],
        ["Cobertura geográfica", `${metrics.paises} países · ${metrics.departamentos} departamentos`],
        ["R² — Agente 3 (XGBoost)",            `${metrics.r2_xgb}  |  MAE ${metrics.mae_xgb}  |  RMSE ${metrics.rmse_xgb}`],
        ["R² — Agente 4 (LSTM PyTorch)",        `${metrics.r2_lstm} |  MAE ${metrics.mae_lstm} |  RMSE ${metrics.rmse_lstm}`],
        ["R² — Ensemble Final (Agentes 3+4+6)", `${metrics.r2_ens}  |  MAE ${metrics.mae}      |  RMSE ${metrics.rmse}`],
      ],
      headStyles: { fillColor: PRIMARY },
      alternateRowStyles: { fillColor: [245, 248, 255] },
    });

    // Top departamentos
    let topDepts = [];
    try {
      const r = await fetch(`${API_URL}/api/top-departments?n=10`);
      if (r.ok) topDepts = await r.json();
    } catch (_) {}

    const y1 = doc.lastAutoTable?.finalY ?? 100;
    doc.setFontSize(12);
    doc.setTextColor(...PRIMARY);
    doc.text("2. Top 10 Focos de Mayor Incidencia Histórica", 14, y1 + 10);
    autoTable(doc, {
      startY: y1 + 14,
      head: [["Departamento", "País", "Incidencia media (casos/100k)", "Máximo registrado"]],
      body: topDepts.length > 0
        ? topDepts.map((d) => [
            d.adm_1_name ?? "—",
            d.pais ?? "—",
            (d.mean_incidencia ?? "—").toString(),
            (d.max_incidencia  ?? "—").toString(),
          ])
        : [["Sin datos disponibles", "", "", ""]],
      headStyles: { fillColor: ORANGE },
      alternateRowStyles: { fillColor: [255, 251, 235] },
    });

    // Arquitectura del sistema
    const y2 = doc.lastAutoTable?.finalY ?? 160;
    doc.setFontSize(12);
    doc.setTextColor(...PRIMARY);
    doc.text("3. Arquitectura Multi-Agente", 14, y2 + 10);
    autoTable(doc, {
      startY: y2 + 14,
      head: [["Agente", "Rol", "Tecnología"]],
      body: [
        ["Agente 1", "Ingesta de datos",                  "OpenDengue + NASA POWER API"],
        ["Agente 2", "Preprocesamiento",                  "Pandas, NumPy, Scikit-Learn (StandardScaler)"],
        ["Agente 3", "Predicción ML",                     "XGBoost + SHAP (TreeSHAP) · R²=91.49%"],
        ["Agente 4", "Predicción DL (series temporales)", "LSTM PyTorch · 2 capas · 12-mes lookback · R²=90.35%"],
        ["Agente 5", "Consenso Ensemble",                 "Ponderación adaptativa (w=0.50/0.50) + percentiles calibrados · R²=91.47%"],
        ["Agente 6", "Detección de Régimen Epidémico",   "Percentiles locales p25/p50/p90 + ajuste dinámico de pesos ensemble"],
      ],
      headStyles: { fillColor: [79, 70, 229] },
      alternateRowStyles: { fillColor: [245, 245, 255] },
    });

    // Niveles de alerta
    const y3 = doc.lastAutoTable?.finalY ?? 220;
    doc.setFontSize(12);
    doc.setTextColor(...PRIMARY);
    doc.text("4. Niveles de Alerta Epidemiológica", 14, y3 + 10);
    autoTable(doc, {
      startY: y3 + 14,
      head: [["Nivel", "Umbral (percentil histórico)", "Acción recomendada"]],
      body: [
        ["Normal",     "< P25",      "Monitoreo rutinario"],
        ["Vigilancia", "P25 – P50",  "Refuerzo de vigilancia pasiva"],
        ["Alerta",     "P50 – P90",  "Activación de equipos de respuesta"],
        ["Epidemia",   "> P90",      "Declaración de emergencia sanitaria"],
      ],
      headStyles: { fillColor: [16, 185, 129] },
    });

    // Footer
    const pageH = doc.internal.pageSize.getHeight();
    doc.setFontSize(8);
    doc.setTextColor(...GRAY);
    doc.text("DenguePredict — Proyecto Final FISI-UNMSM | Uso académico", 14, pageH - 8);

    doc.save("DenguePredict_ReporteCompleto.pdf");
  };

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
            onSimulationComplete={(sim) =>
              setSimulationHistory((prev) => [{ ...sim, id: Date.now(), timestamp: new Date() }, ...prev])
            }
          />
        );
      case "explain":
        return <ExplainabilityView activeSubtab={activeSubtab} simulationHistory={simulationHistory} onClearHistory={() => setSimulationHistory([])} />;
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
      {/* PANTALLA DE CARGA IMPERIAL (SPLASH SCREEN) */}
      <div
        className={`fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background text-on-background transition-all duration-700 ease-in-out ${
          !isAppLoading ? "opacity-0 pointer-events-none translate-y-[-20px]" : "opacity-100"
        }`}
      >
        {/* Halo de brillo de fondo */}
        <div className="absolute w-[300px] h-[300px] rounded-full bg-primary/10 dark:bg-primary/20 blur-[100px] animate-pulse pointer-events-none" />

        <div className="relative flex flex-col items-center gap-md text-center max-w-sm px-lg z-10">
          {/* Mosquito Grande Animado */}
          <div className="relative p-xl rounded-full bg-surface-container-low/80 dark:bg-zinc-900/60 border border-outline-variant shadow-2xl animate-float">
            <MosquitoIcon size={76} className="text-primary animate-pulse" />
            <span className="absolute top-2 right-2 flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-primary"></span>
            </span>
          </div>

          {/* Nombre del Sistema */}
          <div className="space-y-xs animate-fade-in-up mt-sm">
            <h1 className="text-[32px] font-black tracking-tight text-on-background flex items-center justify-center gap-xs">
              Dengue<span className="text-primary">Predict</span>
            </h1>
            <p className="text-[12px] text-on-surface-variant font-semibold tracking-wider uppercase">
              Sistema Multi-Agente SMA-ML/DL
            </p>
            <p className="text-[11px] text-on-surface-variant italic max-w-[280px] mx-auto">
              Vigilancia Epidemiológica y Alerta Temprana a Nivel Subnacional
            </p>
          </div>

          {/* Barra de progreso y carga */}
          <div className="w-48 mt-lg space-y-xs animate-fade-in-up delay-150">
            <div className="h-[3px] w-full bg-surface-container-high rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full animate-shimmer" style={{
                width: '100%',
                backgroundImage: 'linear-gradient(90deg, rgb(var(--color-primary)) 0%, #3b82f6 50%, rgb(var(--color-primary)) 100%)',
                backgroundSize: '200% 100%'
              }} />
            </div>
            <div className="flex justify-between items-center text-[10px] text-on-surface-variant font-semibold uppercase px-0.5">
              <span>Inicializando agentes</span>
              <span className="animate-pulse">Conectando...</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar — fixed, hidden on mobile */}
      <Sidebar
        currentView={currentView}
        setCurrentView={setCurrentView}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenSupport={() => setIsSupportOpen(true)}
        onGenerateReport={handleGenerateReport}
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
        <section className="flex-1 overflow-y-auto px-sm sm:px-lg pt-md sm:pt-lg pb-32 md:pb-xl">
          <div key={currentView} className="max-w-[1440px] mx-auto animate-fade-in-up">
            {renderView()}
          </div>
        </section>

        {/* Footer — oculto en móvil para no competir con bottom nav */}
        <footer className="hidden md:flex border-t border-outline-variant px-lg py-md flex-col sm:flex-row justify-between items-center gap-md text-on-surface-variant text-label-md">
          <p>© {new Date().getFullYear()} DenguePredict — Unidad de Análisis SMA-ML/DL</p>
          <div className="flex gap-lg">
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Documentación</span>
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Glosario</span>
            <span className="hover:text-primary transition-colors cursor-pointer" onClick={() => setIsSupportOpen(true)}>Soporte</span>
          </div>
        </footer>
      </main>

      {/* Mobile Bottom Nav */}
      <BottomNav currentView={currentView} setCurrentView={setCurrentView} />

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
                  <li><strong>R² (Coeficiente de Determinación):</strong> Porcentaje de variabilidad de los datos explicado por el modelo. A mayor R², mayor fidelidad predictiva (91.47% para el Ensemble XGBoost + LSTM).</li>
                  <li><strong>MAE (Error Absoluto Medio):</strong> Promedio de las desviaciones absolutas entre la predicción y el caso real (5.83 casos/100k hab. para el Ensemble).</li>
                </ul>
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
