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
  const [stats, setStats] = useState({ records: "16,224", r2: "89.79%" });
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
          ["XGBoost R²", metrics?.r2_xgb != null ? `${(metrics.r2_xgb * 100).toFixed(2)}%` : "91.23%"],
          ["LSTM PyTorch R²", metrics?.r2_lstm != null ? `${(metrics.r2_lstm * 100).toFixed(2)}%` : "86.94%"],
          ["MAE Ensemble", metrics?.mae_ensemble != null ? `${metrics.mae_ensemble.toFixed(2)} casos/100k` : "5.97 casos/100k"],
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
          Vigilancia epidemiológica de la incidencia de dengue en América Latina. Seleccione un departamento en el mapa para iniciar una simulación predictiva.
        </p>
      </div>

      {/* KPI Row — 4 tarjetas */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-lg">
        {[
          {
            icon: "database", iconGrad: "from-blue-600 to-primary", iconColor: "text-white",
            label: "Registros", value: stats.records, sub: "observaciones",
            topBar: "bg-gradient-to-r from-primary/60 to-primary/20",
            badge: "ML", delay: "delay-0",
          },
          {
            icon: "calendar_today", iconGrad: "from-emerald-500 to-secondary", iconColor: "text-white",
            label: "Rango temporal", value: "2014–2022", sub: "Train 2014–20 · Test 2021–22",
            topBar: "bg-gradient-to-r from-secondary/60 to-secondary/20",
            badge: "9 años", delay: "delay-75",
          },
          {
            icon: "public", iconGrad: "from-emerald-500 to-teal-600", iconColor: "text-white",
            label: "Países", value: metrics?.n_paises ?? 8, sub: "América Latina",
            topBar: "bg-gradient-to-r from-emerald-500/60 to-emerald-500/20",
            badge: "LATAM", delay: "delay-150",
          },
          {
            icon: "location_city", iconGrad: "from-indigo-500 to-violet-600", iconColor: "text-white",
            label: "Departamentos", value: metrics?.n_departamentos ?? 169, sub: "subregiones activas",
            topBar: "bg-gradient-to-r from-indigo-500/60 to-indigo-500/20",
            badge: "subregiones", delay: "delay-225",
          },
        ].map((kpi) => (
          <div
            key={kpi.label}
            className={`group card-hover bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant rounded-xl overflow-hidden animate-fade-in-up ${kpi.delay} relative`}
          >
            <div className={`h-1 w-full ${kpi.topBar}`} />
            {/* Mobile: columna centrada · sm+: fila */}
            <div className="p-md sm:p-lg flex flex-col sm:flex-row items-center sm:items-center gap-sm sm:gap-md">
              <div className={`w-11 h-11 sm:w-14 sm:h-14 bg-gradient-to-br ${kpi.iconGrad} rounded-xl flex items-center justify-center ${kpi.iconColor} flex-shrink-0 shadow-md transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}>
                <span className="material-symbols-outlined text-[22px] sm:text-[28px]" style={{ fontVariationSettings: "'FILL' 1" }}>{kpi.icon}</span>
              </div>
              <div className="flex-1 min-w-0 text-center sm:text-left">
                <p className="text-[10px] sm:text-[11px] font-semibold text-on-surface-variant uppercase tracking-wide">{kpi.label}</p>
                <h3 className="text-[20px] sm:text-[26px] font-black text-primary leading-tight" style={{ fontVariantNumeric: "tabular-nums" }}>{kpi.value}</h3>
                <p className="text-[9px] sm:text-[10px] text-on-surface-variant truncate">{kpi.sub}</p>
              </div>
              <span className="absolute top-3 right-2 text-[9px] font-bold uppercase tracking-wider text-on-surface-variant/40 hidden group-hover:block">
                {kpi.badge}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Model Performance Row — 5 tarjetas de métricas */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-md sm:gap-lg">
        {[
          {
            label: "Ensemble R²",
            value: metrics?.r2_ensemble != null ? `${(metrics.r2_ensemble * 100).toFixed(2)}%` : "91.47%",
            sub: "log1p · test 2021–22",
            icon: "hub",
            accent: "border-t-2 border-t-primary",
            ring: "ring-1 ring-primary/20",
            iconGrad: "from-blue-600 to-primary",
          },
          {
            label: "XGBoost R²",
            value: metrics?.r2_xgb != null ? `${(metrics.r2_xgb * 100).toFixed(2)}%` : "91.49%",
            sub: "escala log1p",
            icon: "precision_manufacturing",
            accent: "border-t-2 border-t-orange-500",
            ring: "ring-1 ring-orange-200 dark:ring-orange-900/40",
            iconGrad: "from-orange-400 to-orange-600",
          },
          {
            label: "LSTM R²",
            value: metrics?.r2_lstm != null ? `${(metrics.r2_lstm * 100).toFixed(2)}%` : "90.35%",
            sub: "escala log1p",
            icon: "neurology",
            accent: "border-t-2 border-t-purple-500",
            ring: "ring-1 ring-purple-200 dark:ring-purple-900/40",
            iconGrad: "from-purple-500 to-violet-600",
          },
          {
            label: "MAE Ensemble",
            value: metrics?.mae_ensemble != null ? metrics.mae_ensemble.toFixed(2) : "5.83",
            sub: "casos/100k hab.",
            icon: "straighten",
            accent: "border-t-2 border-t-emerald-500",
            ring: "ring-1 ring-emerald-200 dark:ring-emerald-900/40",
            iconGrad: "from-emerald-400 to-teal-600",
          },
          {
            label: "RMSE Ensemble",
            value: metrics?.rmse_ensemble != null ? metrics.rmse_ensemble.toFixed(2) : "20.80",
            sub: "casos/100k hab.",
            icon: "query_stats",
            accent: "border-t-2 border-t-sky-500",
            ring: "ring-1 ring-sky-200 dark:ring-sky-900/40",
            iconGrad: "from-sky-400 to-cyan-600",
          },
        ].map((m, i) => (
          <div
            key={m.label}
            className={`group card-hover bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant ${m.accent} ${m.ring} p-sm sm:p-md rounded-xl animate-fade-in-up ${i === 4 ? "col-span-2 sm:col-span-1" : ""}`}
            style={{ animationDelay: `${300 + i * 75}ms` }}
          >
            <div className={`flex items-center gap-sm mb-xs sm:mb-sm ${i === 4 ? "sm:flex-col sm:items-start md:flex-row" : ""}`}>
              <div className={`w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br ${m.iconGrad} rounded-lg flex items-center justify-center text-white flex-shrink-0 transition-all duration-300 group-hover:scale-110 group-hover:shadow-md`}>
                <span className="material-symbols-outlined text-[14px] sm:text-[16px]" style={{ fontVariationSettings: "'FILL' 1" }}>{m.icon}</span>
              </div>
              <p className="text-[10px] font-semibold text-on-surface-variant uppercase tracking-wide leading-tight">{m.label}</p>
            </div>
            <h3 className="text-[18px] sm:text-[22px] font-black text-primary leading-none mb-xs" style={{ fontVariantNumeric: "tabular-nums" }}>{m.value}</h3>
            <p className="text-[9px] sm:text-[10px] text-on-surface-variant">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-12 gap-lg">
        {/* Map Panel */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-md">
          <div className="bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant rounded-xl overflow-hidden flex flex-col relative transition-shadow duration-200 hover:shadow-xl animate-fade-in-up delay-300" style={{ height: "600px" }}>
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
          <div className="card-hover bg-white dark:bg-zinc-900 dark:border-zinc-800 border border-outline-variant p-lg rounded-xl flex-1 flex flex-col justify-between animate-fade-in-up delay-375 group">
            <div className="flex justify-between items-start mb-lg">
              <div>
                <h4 className="text-headline-md font-bold text-primary">Resumen de Incidencia</h4>
                <p className="text-label-md text-on-surface-variant">Mayores focos registrados (Histórico)</p>
              </div>
              <span className="material-symbols-outlined text-primary/40 text-[28px] transition-all duration-300 group-hover:text-primary group-hover:scale-110"
                style={{ fontVariationSettings: "'FILL' 1" }}>bar_chart</span>
            </div>

            <div className="flex-grow flex flex-col justify-around gap-md">
              {topDepts.map((dept, idx) => (
                <div key={idx} className="space-y-xs group/bar cursor-default">
                  <div className="flex justify-between text-label-md font-medium">
                    <span className="transition-colors group-hover/bar:text-primary">{dept.name}</span>
                    <span className="font-bold tabular" style={{ fontVariantNumeric: "tabular-nums" }}>{dept.value}</span>
                  </div>
                  <div className="h-2.5 w-full bg-surface-container rounded-full overflow-hidden">
                    <div
                      className={`chart-bar h-full ${dept.color} rounded-full transition-all duration-300 group-hover/bar:brightness-110`}
                      style={{ width: `${dept.pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleDownloadReport}
              className="btn-primary mt-lg w-full py-3 border border-primary text-primary rounded-xl font-bold text-label-md
                hover:bg-primary hover:text-on-primary transition-all duration-200 flex items-center justify-center gap-md cursor-pointer group/btn"
            >
              Descargar Reporte PDF
              <span className="material-symbols-outlined text-[18px] transition-transform duration-200 group-hover/btn:translate-y-0.5"
                style={{ fontVariationSettings: "'FILL' 1" }}>download</span>
            </button>
          </div>

          {/* Model Status Card */}
          <div className="relative overflow-hidden bg-primary text-on-primary p-lg rounded-xl shadow-xl transition-all duration-200 hover:shadow-2xl hover:brightness-105 animate-fade-in-up delay-450 group">
            {/* Decorative bg circle */}
            <div className="absolute -right-8 -top-8 w-32 h-32 rounded-full bg-white/5 transition-transform duration-500 group-hover:scale-125" />
            <div className="absolute -right-2 -bottom-6 w-20 h-20 rounded-full bg-white/5 transition-transform duration-500 group-hover:scale-110" />
            <div className="relative z-10">
              <div className="flex items-center gap-md mb-md">
                <span className="material-symbols-outlined text-secondary-fixed text-[24px] transition-transform duration-300 group-hover:scale-110"
                  style={{ fontVariationSettings: "'FILL' 1" }}>psychology</span>
                <h5 className="text-label-md font-bold uppercase tracking-wider">Estado del Modelo Híbrido</h5>
              </div>
              <p className="text-body-md opacity-90 leading-relaxed">
                El motor ensemble combina <strong>XGBoost</strong> (R²={metrics?.r2_xgb != null ? `${(metrics.r2_xgb * 100).toFixed(1)}%` : "91.5%"}) y <strong>LSTM PyTorch</strong> (R²={metrics?.r2_lstm != null ? `${(metrics.r2_lstm * 100).toFixed(1)}%` : "90.4%"}) alcanzando <strong>R²={metrics?.r2_ensemble != null ? `${(metrics.r2_ensemble * 100).toFixed(1)}%` : "91.5%"}</strong> y MAE de {metrics?.mae_ensemble != null ? `${metrics.mae_ensemble.toFixed(2)}` : "5.83"} casos/100k hab.
              </p>
              <div className="mt-md flex items-center gap-sm flex-wrap">
                {["XGBoost", "LSTM PyTorch", "FastAPI In-Memory"].map((tag) => (
                  <span key={tag} className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full hover:bg-white/20 transition-colors cursor-default">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scatter Plot */}
      {scatterData !== undefined && (
        <ScatterPlot data={scatterData} darkMode={darkMode} metrics={metrics} />
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
