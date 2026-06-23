import React, { useEffect, useState } from "react";
import Map from "./MapContainer";
import ScatterPlot from "./ScatterPlot";
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
  const [stats, setStats] = useState({ records: "9,600", r2: "89.79%" });
  const [metrics, setMetrics] = useState(null);
  const [scatterData, setScatterData] = useState(undefined); // undefined=loading, null=unavailable
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
          setMetrics(m);
          setStats({
            records: m.records_procesados?.toLocaleString() ?? "9,600",
            r2: m.r2_ensemble != null ? `${(m.r2_ensemble * 100).toFixed(2)}%` : (m.r2_xgb != null ? `${(m.r2_xgb * 100).toFixed(2)}%` : "—"),
          });
        }
        // Load scatter data (may not exist yet)
        try {
          const scatterRes = await fetch(`${backendUrl}/api/scatter-data`);
          setScatterData(scatterRes.ok ? await scatterRes.json() : null);
        } catch {
          setScatterData(null);
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
          ["XGBoost R²", "91.23%"],
          ["LSTM PyTorch R²", "86.94%"],
          ["MAE Ensemble", "5.97 casos/100k"],
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

      {/* KPI Row — 4 tarjetas */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-lg">
        {/* Registros */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex items-center gap-md shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-12 h-12 bg-surface-container rounded-lg flex items-center justify-center text-primary flex-shrink-0">
            <span className="material-symbols-outlined text-[28px]">database</span>
          </div>
          <div>
            <p className="text-[11px] font-medium text-on-surface-variant">Registros</p>
            <h3 className="text-[22px] font-bold text-primary leading-tight" style={{ fontVariantNumeric: "tabular-nums" }}>
              {stats.records}
            </h3>
            <p className="text-[10px] text-on-surface-variant">observaciones</p>
          </div>
        </div>

        {/* Rango */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant border-l-4 border-l-secondary p-lg rounded-xl flex items-center gap-md shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-12 h-12 bg-secondary-container/20 rounded-lg flex items-center justify-center text-secondary flex-shrink-0">
            <span className="material-symbols-outlined text-[28px]">calendar_today</span>
          </div>
          <div>
            <p className="text-[11px] font-medium text-on-surface-variant">Rango temporal</p>
            <h3 className="text-[22px] font-bold text-primary leading-tight" style={{ fontVariantNumeric: "tabular-nums" }}>2014–2022</h3>
            <p className="text-[10px] text-on-surface-variant">
              Train: 2014–2020 · Test: 2021–2022
            </p>
          </div>
        </div>

        {/* Países */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex items-center gap-md shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-12 h-12 bg-emerald-50 rounded-lg flex items-center justify-center text-emerald-600 flex-shrink-0">
            <span className="material-symbols-outlined text-[28px]">public</span>
          </div>
          <div>
            <p className="text-[11px] font-medium text-on-surface-variant">Países</p>
            <h3 className="text-[22px] font-bold text-primary leading-tight">{metrics?.n_paises ?? 8}</h3>
            <p className="text-[10px] text-on-surface-variant">América Latina</p>
          </div>
        </div>

        {/* Departamentos */}
        <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex items-center gap-md shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
          <div className="w-12 h-12 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 flex-shrink-0">
            <span className="material-symbols-outlined text-[28px]">location_city</span>
          </div>
          <div>
            <p className="text-[11px] font-medium text-on-surface-variant">Departamentos</p>
            <h3 className="text-[22px] font-bold text-primary leading-tight">{metrics?.n_departamentos ?? 169}</h3>
            <p className="text-[10px] text-on-surface-variant">subregiones</p>
          </div>
        </div>
      </div>

      {/* Model Performance Row — 5 tarjetas de métricas del modelo */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-lg">
        {[
          {
            label: "Ensemble R²",
            value: metrics?.r2_ensemble != null ? `${(metrics.r2_ensemble * 100).toFixed(2)}%` : "89.79%",
            sub: "escala log1p · test 2021–22",
            icon: "hub",
            color: "text-primary",
            bg: "bg-primary-container/10",
          },
          {
            label: "XGBoost R²",
            value: metrics?.r2_xgb != null ? `${(metrics.r2_xgb * 100).toFixed(2)}%` : "91.23%",
            sub: "escala log1p",
            icon: "precision_manufacturing",
            color: "text-orange-600",
            bg: "bg-orange-50",
          },
          {
            label: "LSTM R²",
            value: metrics?.r2_lstm != null ? `${(metrics.r2_lstm * 100).toFixed(2)}%` : "86.94%",
            sub: "escala log1p",
            icon: "neurology",
            color: "text-purple-600",
            bg: "bg-purple-50",
          },
          {
            label: "MAE Ensemble",
            value: metrics?.mae_ensemble != null ? metrics.mae_ensemble.toFixed(2) : "5.97",
            sub: "casos/100k hab.",
            icon: "straighten",
            color: "text-emerald-600",
            bg: "bg-emerald-50",
          },
          {
            label: "RMSE Ensemble",
            value: metrics?.rmse_ensemble != null ? metrics.rmse_ensemble.toFixed(2) : "21.24",
            sub: "casos/100k hab.",
            icon: "query_stats",
            color: "text-sky-600",
            bg: "bg-sky-50",
          },
        ].map((m) => (
          <div key={m.label} className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-md rounded-xl flex items-center gap-md shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
            <div className={`w-10 h-10 ${m.bg} rounded-lg flex items-center justify-center ${m.color} flex-shrink-0`}>
              <span className="material-symbols-outlined text-[22px]">{m.icon}</span>
            </div>
            <div>
              <p className="text-[11px] font-medium text-on-surface-variant">{m.label}</p>
              <h3 className="text-[20px] font-bold text-primary leading-tight" style={{ fontVariantNumeric: "tabular-nums" }}>{m.value}</h3>
              <p className="text-[10px] text-on-surface-variant">{m.sub}</p>
            </div>
          </div>
        ))}
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
              El motor ensemble combina <strong>XGBoost</strong> (R²=91.2%) y <strong>LSTM PyTorch</strong> (R²=86.9%) con pesos proporcionales al R², alcanzando <strong>R²=89.8%</strong> y MAE de 5.97 casos/100k hab. sobre el set de prueba 2021–2022.
            </p>
            <div className="mt-md flex items-center gap-sm flex-wrap">
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">XGBoost</span>
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">LSTM PyTorch</span>
              <span className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">FastAPI In-Memory</span>
            </div>
          </div>
        </div>
      </div>

      {/* Scatter Plot */}
      {scatterData !== undefined && (
        <ScatterPlot data={scatterData} />
      )}

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
