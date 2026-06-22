import React, { useEffect, useState } from "react";
import Map from "./MapContainer";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const countryNames = {
  COL: "Colombia",
  BRA: "Brasil",
  MEX: "México",
  PER: "Perú",
  ARG: "Argentina",
  BOL: "Bolivia",
  ECU: "Ecuador",
  PAN: "Panamá",
  SLV: "El Salvador",
  HND: "Honduras",
  GTM: "Guatemala",
  CRI: "Costa Rica",
  DOM: "República Dominicana",
  VEN: "Venezuela",
  PRY: "Paraguay",
  CHL: "Chile",
  URY: "Uruguay"
};

export default function DashboardView({ coordinates, onSelectDepartment, backendUrl, darkMode }) {
  const BAR_COLORS = [
    "bg-gradient-to-r from-red-500 to-red-400",
    "bg-gradient-to-r from-orange-500 to-orange-400",
    "bg-gradient-to-r from-orange-400 to-yellow-500",
    "bg-gradient-to-r from-yellow-500 to-yellow-400",
    "bg-gradient-to-r from-yellow-400 to-yellow-300",
  ];
  const [topDepts, setTopDepts] = useState([]);

  const [backendReady, setBackendReady] = useState(false);
  const [stats, setStats] = useState({ records: "9,600", r2: "88.80%" });
  const [selectedCountryFilter, setSelectedCountryFilter] = useState("ALL");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportProgress, setReportProgress] = useState(0);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/status`);
        if (res.ok) {
          setBackendReady(true);
        }
        // Load real metrics from backend
        const metricsRes = await fetch(`${backendUrl}/api/metrics`);
        if (metricsRes.ok) {
          const m = await metricsRes.json();
          setStats({
            records: m.records_procesados?.toLocaleString() ?? "9,600",
            r2: m.r2_ensemble != null ? `${(m.r2_ensemble * 100).toFixed(2)}%` : (m.r2_xgb != null ? `${(m.r2_xgb * 100).toFixed(2)}%` : "—"),
          });
        }
        // Load real top departments
        const topRes = await fetch(`${backendUrl}/api/top-departments?n=5`);
        if (topRes.ok) {
          const raw = await topRes.json();
          setTopDepts(raw.map((d, i) => ({
            ...d,
            value: d.mean_incidencia,
            color: BAR_COLORS[i % BAR_COLORS.length],
          })));
        }
      } catch (err) {
        console.warn("Backend no disponible para Dashboard", err);
      }
    };
    loadStatus();
  }, [backendUrl]);

  // Extraer los países que realmente existen en las coordenadas
  const uniqueCountries = [...new Set(coordinates.map((c) => c.iso_a0))].filter(Boolean);

  // Filtrar coordenadas según el filtro seleccionado
  const filteredCoords = selectedCountryFilter === "ALL"
    ? coordinates
    : coordinates.filter((c) => c.iso_a0 === selectedCountryFilter);

  const handleDownloadReport = () => {
    setIsGeneratingReport(true);
    setReportProgress(20);

    try {
      const doc = new jsPDF();
      const fecha = new Date().toLocaleDateString("es-ES", { year: "numeric", month: "long", day: "numeric" });

      doc.setFontSize(18);
      doc.setTextColor(30, 58, 95);
      doc.text("DenguePredict", 14, 18);
      doc.setFontSize(11);
      doc.setTextColor(100, 100, 100);
      doc.text("Reporte Epidemiológico — Sistema Multi-Agente (XGBoost + LSTM)", 14, 26);
      doc.text(`Generado: ${fecha}`, 14, 32);

      setReportProgress(40);

      autoTable(doc, {
        startY: 40,
        head: [["Métrica del Sistema", "Valor"]],
        body: [
          ["Registros consolidados", stats.records + " obs."],
          ["Rango temporal", "2014 — 2022"],
          ["Precisión del sistema (R² Ensemble)", stats.r2],
          ["XGBoost R²", "89.88%"],
          ["LSTM PyTorch R²", "85.77%"],
          ["MAE Ensemble", "7.11 casos/100k"],
        ],
        headStyles: { fillColor: [30, 58, 95] },
        alternateRowStyles: { fillColor: [245, 248, 255] },
      });

      setReportProgress(70);

      const finalY = doc.lastAutoTable?.finalY ?? 100;
      doc.setFontSize(13);
      doc.setTextColor(30, 58, 95);
      doc.text("Top 5 Focos de Mayor Incidencia Histórica", 14, finalY + 12);

      autoTable(doc, {
        startY: finalY + 16,
        head: [["Departamento", "País", "Media (casos/100k)", "Máximo (casos/100k)"]],
        body: topDepts.map((d) => [
          d.name ?? d.adm_1_name ?? "—",
          d.pais ?? "—",
          (d.mean_incidencia ?? d.value ?? "—").toString(),
          (d.max_incidencia ?? "—").toString(),
        ]),
        headStyles: { fillColor: [217, 119, 6] },
        alternateRowStyles: { fillColor: [255, 251, 235] },
      });

      setReportProgress(95);
      doc.save("DenguePredict_Reporte.pdf");
    } catch (err) {
      console.error("Error generando PDF:", err);
    } finally {
      setReportProgress(100);
      setTimeout(() => setIsGeneratingReport(false), 600);
    }
  };

  return (
    <div className="max-w-[1440px] mx-auto space-y-lg">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-sm mb-xs">
          <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>monitoring</span>
          <span className="text-secondary text-label-md font-bold uppercase tracking-wider">Vista General del Sistema</span>
        </div>
        <h2 className="text-headline-lg text-primary font-bold">Panel de Control Epidemiológico</h2>
        <p className="text-on-surface-variant text-body-md mt-xs max-w-2xl">
          Monitoreo en tiempo real de la incidencia de dengue en América Latina. Seleccione un departamento en el mapa para iniciar una simulación predictiva.
        </p>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-lg">
        {/* Registros */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex items-center gap-lg shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-14 h-14 bg-surface-container rounded-lg flex items-center justify-center text-primary">
            <span className="material-symbols-outlined text-[32px]">database</span>
          </div>
          <div>
            <p className="text-label-md font-medium text-on-surface-variant">
              Registros Consolidados
            </p>
            <h3 className="text-headline-md font-bold text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>
              {stats.records} <span className="text-label-md font-normal opacity-60">obs.</span>
            </h3>
          </div>
        </div>

        {/* Rango */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant border-l-4 border-l-secondary p-lg rounded-xl flex items-center gap-lg shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-14 h-14 bg-secondary-container/20 rounded-lg flex items-center justify-center text-secondary">
            <span className="material-symbols-outlined text-[32px]">calendar_today</span>
          </div>
          <div>
            <p className="text-label-md font-medium text-on-surface-variant">Rango Temporal</p>
            <h3 className="text-headline-md font-bold text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>
              2014 — 2022
            </h3>
          </div>
        </div>

        {/* Precisión */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex items-center gap-lg shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-14 h-14 bg-primary-container/10 rounded-lg flex items-center justify-center text-primary-container">
            <span className="material-symbols-outlined text-[32px]">analytics</span>
          </div>
          <div>
            <p className="text-label-md font-medium text-on-surface-variant">Precisión del Sistema</p>
            <h3 className="text-headline-md font-bold text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>
              {stats.r2} <span className="text-label-md font-normal text-secondary">(R²)</span>
            </h3>
          </div>
        </div>
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-12 gap-lg">
        {/* Map Panel */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-md">
          <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant rounded-xl overflow-hidden flex flex-col relative shadow-[0px_4px_20px_rgba(30,58,95,0.04)]" style={{ height: "600px" }}>
            <div className="p-md bg-white/80 dark:bg-zinc-900/80 backdrop-blur-sm z-10 border-b border-outline-variant flex flex-wrap gap-md justify-between items-center">
              <h4 className="text-label-md font-bold text-primary uppercase tracking-wider">
                Distribución Geoespacial de Riesgo (OpenStreetMap)
              </h4>
              <div className="flex items-center gap-md">
                <div className="flex items-center gap-xs">
                  <span className="text-[12px] font-bold text-primary">Filtrar País:</span>
                  <select
                    value={selectedCountryFilter}
                    onChange={(e) => setSelectedCountryFilter(e.target.value)}
                    className="bg-surface-container-high text-primary font-bold text-[11px] px-sm py-1 rounded-lg border border-outline-variant outline-none cursor-pointer hover:bg-surface-container-highest transition-colors"
                  >
                    <option value="ALL">Todos los países</option>
                    {uniqueCountries.map((c) => (
                      <option key={c} value={c}>
                        {countryNames[c] || c}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-sm">
                  <span className="text-[12px] font-medium text-on-surface-variant hidden sm:inline">
                    Vigilancia Activa
                  </span>
                  <span className="w-2.5 h-2.5 rounded-full bg-secondary animate-pulse"></span>
                </div>
              </div>
            </div>

            {/* Interactive Leaflet Map */}
            <div className="flex-1 relative">
              <Map
                coordinates={filteredCoords}
                onSelectDepartment={onSelectDepartment}
                backendUrl={backendUrl}
                darkMode={darkMode}
              />

              {/* Map Legend */}
              <div
                className="absolute bottom-md left-md bg-white/95 dark:bg-zinc-900/95 backdrop-blur-md p-md rounded-lg border border-outline-variant/30 max-w-[240px] shadow-[0px_4px_20px_rgba(30,58,95,0.04)]"
                style={{ zIndex: 1000 }}
              >
                <p className="text-label-md font-bold text-primary mb-xs">Perfil Endémico Histórico</p>
                <p className="text-[10px] text-on-surface-variant mb-sm">Incidencia media 2014–2022</p>
                <div className="h-2 w-full map-gradient rounded-full mb-xs"></div>
                <div className="flex justify-between text-[10px] font-bold text-on-surface-variant">
                  <span>Bajo</span>
                  <span>Moderado</span>
                  <span>Alto</span>
                  <span>Muy Alto</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel: Summary */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-lg">
          {/* Incidence Summary */}
          <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex-1 flex flex-col justify-between shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
            <div className="flex justify-between items-start mb-lg">
              <div>
                <h4 className="text-headline-md font-bold text-primary">Resumen de Incidencia</h4>
                <p className="text-label-md text-on-surface-variant">Mayores focos registrados (Histórico)</p>
              </div>
              <span className="material-symbols-outlined text-primary-container">bar_chart</span>
            </div>

            <div className="flex-grow flex flex-col justify-around gap-md">
              {topDepts.map((dept, idx) => (
                <div key={idx} className="space-y-xs">
                  <div className="flex justify-between text-label-md font-medium">
                    <span>{dept.name}</span>
                    <span className="font-bold" style={{ fontVariantNumeric: "tabular-nums" }}>{dept.value}</span>
                  </div>
                  <div className="h-2.5 w-full bg-surface-container rounded-full overflow-hidden">
                    <div
                      className={`chart-bar h-full ${dept.color} rounded-full`}
                      style={{ width: `${dept.pct}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleDownloadReport}
              className="mt-lg w-full py-3 border border-primary text-primary rounded-xl font-bold text-label-md hover:bg-primary hover:text-on-primary transition-all flex items-center justify-center gap-md cursor-pointer"
            >
              Descargar Reporte PDF
              <span className="material-symbols-outlined text-[18px]">download</span>
            </button>
          </div>

          {/* Model Status Card */}
          <div className="bg-primary text-on-primary p-lg rounded-xl shadow-xl">
            <div className="flex items-center gap-md mb-md">
              <span className="material-symbols-outlined text-secondary-fixed">psychology</span>
              <h5 className="text-label-md font-bold uppercase tracking-wider">Estado del Modelo Híbrido</h5>
            </div>
            <p className="text-body-md opacity-90 leading-relaxed">
              El motor ensemble combina <strong>XGBoost</strong> (R²=89.9%) y <strong>LSTM PyTorch</strong> (R²=85.8%) con pesos proporcionales al R², alcanzando <strong>R²=88.8%</strong> y MAE de 7.11 casos/100k hab.
            </p>
            <div className="mt-md flex items-center gap-sm flex-wrap">
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">XGBoost</span>
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">LSTM PyTorch</span>
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">FastAPI In-Memory</span>
            </div>
          </div>
        </div>
      </div>

      {/* REPORT GENERATION MODAL */}
      {isGeneratingReport && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[2000] animate-fade-in">
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-2xl p-lg w-full max-w-sm shadow-2xl flex flex-col items-center gap-md text-center">
            <span className="material-symbols-outlined text-primary text-[48px] animate-spin">progress_activity</span>
            <div>
              <h4 className="text-headline-md text-primary font-bold">Generando Reporte PDF...</h4>
              <p className="text-body-md text-on-surface-variant mt-xs">Compilando datos históricos de incidencia</p>
            </div>
            <div className="w-full bg-surface-container rounded-full h-3 overflow-hidden">
              <div 
                className="bg-primary h-full rounded-full transition-all duration-300"
                style={{ width: `${reportProgress}%` }}
              ></div>
            </div>
            <span className="text-label-md font-bold text-primary">{reportProgress}% Completado</span>
          </div>
        </div>
      )}
    </div>
  );
}
