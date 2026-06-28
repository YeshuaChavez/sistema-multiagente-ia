import React, { useEffect, useState, useCallback } from "react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const MOCK_SHAP_GLOBAL = [
  { feature: "incidencia_lag1", importance: 0.285 },
  { feature: "tmax_promedio", importance: 0.182 },
  { feature: "precipitacion", importance: 0.144 },
  { feature: "humedad_promedio", importance: 0.095 },
  { feature: "agua_basica", importance: -0.078 },
  { feature: "incidencia_vecinos_lag1", importance: 0.065 },
  { feature: "densidad_poblacion", importance: 0.042 },
  { feature: "tmin_promedio", importance: 0.038 },
  { feature: "tmax_lag1", importance: 0.031 },
  { feature: "precipitacion_lag1", importance: 0.024 },
];

const MONTH_NAMES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"];

export default function ExplainabilityView({ activeSubtab, simulationHistory = [], onClearHistory }) {
  // ─── Global SHAP state ───
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recalculating, setRecalculating] = useState(false);
  const [shapPage, setShapPage] = useState(0);
  const SHAP_PAGE_SIZE = 15;

  // ─── Local SHAP state ───
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [localResult, setLocalResult] = useState(null);

  const lastSimulation = simulationHistory[selectedIdx] ?? null;

  // Reset result when selected simulation changes
  useEffect(() => { setLocalResult(null); setLocalError(null); }, [selectedIdx]);
  // Always point to newest when a new simulation arrives
  useEffect(() => { setSelectedIdx(0); }, [simulationHistory.length]);

  // ─── Fetch global SHAP ───
  const fetchShap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/explain/global`);
      if (!response.ok) throw new Error("No se pudo cargar la explicabilidad SHAP");
      const raw = await response.json();
      const arr = Object.entries(raw)
        .map(([feature, importance]) => ({ feature, importance }))
        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
      setShapData(arr);
    } catch (err) {
      console.warn("Backend explain offline, usando datos demo...", err.message);
      setShapData(MOCK_SHAP_GLOBAL);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchShap(); }, [fetchShap]);

  const handleAnalyzeLocal = useCallback(async () => {
    if (!lastSimulation) return;
    setLocalLoading(true);
    setLocalError(null);
    setLocalResult(null);
    try {
      const res = await fetch(`${API_URL}/api/predict/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          iso_a0: lastSimulation.iso_a0,
          adm_1_name: lastSimulation.adm_1_name,
          mes: lastSimulation.mes,
          clima_overrides: lastSimulation.clima_overrides, // ya viene sin mes_sin/mes_cos/rolls
          include_shap: true,
        }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      if (!data.shap_local) throw new Error("El backend no retornó valores SHAP locales.");
      const shapArr = Object.entries(data.shap_local)
        .map(([feature, value]) => ({ feature, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 12);
      setLocalResult({
        prediction: data.prediccion_ensemble ?? data.prediccion_ml,
        riesgo: data.riesgo_ensemble ?? data.riesgo_ml,
        shapArr,
      });
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setLocalLoading(false);
    }
  }, [lastSimulation]);

  const handleRecalculate = async () => {
    setRecalculating(true);
    await fetchShap();
    setRecalculating(false);
  };

  const handleExport = () => {
    const doc = new jsPDF();
    const fecha = new Date().toLocaleDateString("es-ES", { year: "numeric", month: "long", day: "numeric" });
    const PRIMARY = [30, 58, 95];
    const GRAY    = [100, 100, 100];

    doc.setFontSize(18);
    doc.setTextColor(...PRIMARY);
    doc.text("DenguePredict — Explicabilidad SHAP", 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(...GRAY);
    doc.text(`Agente 3 · TreeSHAP | Generado: ${fecha}`, 14, 26);

    if (activeSubtab === "Local SHAP") {
      if (!localResult) {
        alert("Primero ejecuta una simulación en el Predictor y luego haz clic en 'Explicar simulación'.");
        return;
      }
      const deptLabel = lastSimulation ? `${lastSimulation.adm_1_name} (${lastSimulation.country})` : "—";
      doc.setFontSize(13);
      doc.setTextColor(...PRIMARY);
      doc.text(`SHAP Local — ${deptLabel}`, 14, 36);
      doc.setFontSize(10);
      doc.setTextColor(...GRAY);
      doc.text(
        `Predicción Ensemble: ${localResult.prediction?.toFixed(2)} casos/100k · Riesgo: ${localResult.riesgo?.nivel ?? "—"}`,
        14, 43
      );
      autoTable(doc, {
        startY: 48,
        head: [["Variable", "Valor SHAP", "Dirección"]],
        body: localResult.shapArr.map((f) => [
          f.feature,
          (f.value > 0 ? "+" : "") + f.value.toFixed(6),
          f.value >= 0 ? "Aumenta riesgo" : "Reduce riesgo",
        ]),
        headStyles: { fillColor: PRIMARY },
        alternateRowStyles: { fillColor: [245, 248, 255] },
        columnStyles: {
          1: { halign: "right", fontStyle: "bold" },
          2: { halign: "center" },
        },
      });
    } else {
      // Global SHAP
      if (!shapData || shapData.length === 0) {
        alert("Los datos SHAP globales aún están cargando.");
        return;
      }
      doc.setFontSize(13);
      doc.setTextColor(...PRIMARY);
      doc.text("SHAP Global — Importancia Media de Variables (todos los departamentos)", 14, 36);
      autoTable(doc, {
        startY: 42,
        head: [["Ranking", "Variable", "Importancia SHAP media", "Dirección"]],
        body: shapData.map((f, i) => [
          `#${i + 1}`,
          f.feature,
          f.importance.toFixed(6),
          f.importance >= 0 ? "Aumenta riesgo" : "Reduce riesgo",
        ]),
        headStyles: { fillColor: PRIMARY },
        alternateRowStyles: { fillColor: [245, 248, 255] },
        columnStyles: {
          2: { halign: "right", fontStyle: "bold" },
          3: { halign: "center" },
        },
      });
    }

    const pageH = doc.internal.pageSize.getHeight();
    doc.setFontSize(8);
    doc.setTextColor(...GRAY);
    doc.text("DenguePredict — Proyecto Final FISI-UNMSM | Uso académico", 14, pageH - 8);

    const filename = activeSubtab === "Local SHAP"
      ? `SHAP_Local_${localDept.replace(/\s+/g, "_")}.pdf`
      : "SHAP_Global_DenguePredict.pdf";
    doc.save(filename);
  };

  const maxVal = Array.isArray(shapData) && shapData.length > 0 
    ? Math.max(...shapData.map((f) => Math.abs(f.importance))) 
    : 1;

  // Timestamp for display
  const now = new Date();
  const hours12 = now.getHours() % 12 || 12;
  const ampm = now.getHours() >= 12 ? "PM" : "AM";
  const timeStr = `Hoy, ${hours12.toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")} ${ampm}`;

  return (
    <div className="max-w-[1440px] mx-auto text-on-surface">
      {/* Header Section */}
      <div className="mb-lg flex flex-col md:flex-row md:items-end justify-between gap-md">
        <div>
          <h1 className="text-headline-lg text-primary font-bold mb-xs">
            Módulo de Explicabilidad Local y Global (SHAP)
          </h1>
          <div className="flex items-center gap-sm flex-wrap">
            <span className="px-md py-xs bg-secondary/10 text-secondary text-label-md font-medium rounded-full flex items-center gap-xs">
              <span className="h-2 w-2 rounded-full bg-secondary text-primary"></span> Agente 3 — TreeSHAP
            </span>
            <span className="text-on-surface-variant text-label-md">Última actualización: {timeStr}</span>
          </div>
        </div>
        <div className="flex gap-md">
          <button 
            onClick={handleExport}
            className="px-md py-sm border border-primary text-primary rounded-lg text-label-md font-medium hover:bg-primary/5 transition-colors flex items-center gap-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">download</span> Exportar PDF
          </button>
          <button 
            onClick={handleRecalculate}
            disabled={recalculating}
            className="px-md py-sm bg-primary text-on-primary rounded-lg text-label-md font-medium hover:bg-primary-container transition-colors flex items-center gap-sm cursor-pointer disabled:opacity-55"
          >
            <span className="material-symbols-outlined text-[18px] animate-pulse">refresh</span> 
            {recalculating ? "Procesando..." : "Re-calcular SHAP"}
          </button>
        </div>
      </div>

      {/* Bento Grid */}
      <div className="grid grid-cols-1 gap-lg mb-lg">

        {/* ═══ TAB CONTENT: GLOBAL SHAP Summary Plot ═══ */}
        {activeSubtab === "Global SHAP" && (
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] max-w-4xl mx-auto w-full animate-fade-in">
            <div className="flex items-center justify-between mb-lg">
              <h3 className="text-headline-md text-on-surface font-bold">SHAP Global Summary Plot</h3>
              <span
                className="material-symbols-outlined text-on-surface-variant cursor-help"
                title="Impacto global de las variables en el modelo a nivel de todo el continente"
              >
                info
              </span>
            </div>
            <p className="text-on-surface-variant text-label-md mb-xl">
              Importancia media de las características (Magnitud del valor SHAP promedio de los agentes ML/DL)
            </p>

            {/* Loading */}
            {loading && (
              <div className="space-y-lg">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="space-y-sm">
                    <div className="flex justify-between">
                      <div className="h-4 w-28 shimmer rounded"></div>
                      <div className="h-4 w-12 shimmer rounded"></div>
                    </div>
                    <div className="w-full h-4 shimmer rounded-full"></div>
                  </div>
                ))}
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="bg-error-container p-md rounded-lg flex items-center gap-md">
                <span className="material-symbols-outlined text-on-error-container">error</span>
                <div>
                  <p className="text-label-md text-on-error-container font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* SHAP Bars */}
            {Array.isArray(shapData) && shapData.length > 0 && (() => {
              const totalPages = Math.ceil(shapData.length / SHAP_PAGE_SIZE);
              const pageData  = shapData.slice(shapPage * SHAP_PAGE_SIZE, (shapPage + 1) * SHAP_PAGE_SIZE);
              return (
                <div className="space-y-lg">
                  {pageData.map((feature, i) => {
                    const globalRank = shapPage * SHAP_PAGE_SIZE + i + 1;
                    const pct = (Math.abs(feature.importance) / maxVal) * 100;
                    const isNegative = feature.importance < 0;
                    return (
                      <div key={feature.feature} className="space-y-xs group/bar">
                        <div className="flex justify-between items-center" style={{ fontVariantNumeric: "tabular-nums" }}>
                          <div className="flex items-center gap-sm">
                            <span className="text-[10px] font-bold text-on-surface-variant/50 w-5 text-right flex-shrink-0">#{globalRank}</span>
                            <span className="text-label-md text-on-surface font-medium group-hover/bar:text-primary transition-colors">{feature.feature}</span>
                          </div>
                          <div className="flex items-center gap-sm">
                            <span className={`text-[10px] font-bold px-xs py-0.5 rounded ${isNegative ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300" : "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300"}`}>
                              {isNegative ? "↓ reduce" : "↑ aumenta"}
                            </span>
                            <span className="text-label-md text-on-surface-variant font-mono">{feature.importance.toFixed(4)}</span>
                          </div>
                        </div>
                        <div className="w-full bg-surface-container-low dark:bg-zinc-800 h-3 rounded-full overflow-hidden">
                          <div
                            className={`chart-bar h-full bg-gradient-to-r ${isNegative ? "from-blue-600 to-blue-400" : "from-orange-400 to-orange-600"} rounded-full transition-all duration-300 group-hover/bar:brightness-110`}
                            style={{ width: `${Math.max(pct, 2)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between pt-lg border-t border-outline-variant mt-lg">
                      <button
                        onClick={() => setShapPage(p => p - 1)}
                        disabled={shapPage === 0}
                        className="flex items-center gap-xs text-label-md text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[18px]">chevron_left</span>
                        Anterior
                      </button>
                      <div className="flex items-center gap-xs">
                        {[...Array(totalPages)].map((_, pi) => (
                          <button
                            key={pi}
                            onClick={() => setShapPage(pi)}
                            className={`w-7 h-7 rounded-lg text-[12px] font-bold transition-colors cursor-pointer
                              ${pi === shapPage ? "bg-primary text-on-primary" : "text-on-surface-variant hover:bg-surface-container"}`}
                          >
                            {pi + 1}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={() => setShapPage(p => p + 1)}
                        disabled={shapPage >= totalPages - 1}
                        className="flex items-center gap-xs text-label-md text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors cursor-pointer"
                      >
                        Siguiente
                        <span className="material-symbols-outlined text-[18px]">chevron_right</span>
                      </button>
                    </div>
                  )}

                  {/* Legend */}
                  <div className="flex justify-between items-center border-t border-outline-variant pt-md">
                    <div className="flex items-center gap-sm">
                      <div className="w-3 h-3 rounded-full bg-blue-500" />
                      <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Atenuación del Riesgo (SHAP &lt; 0)</span>
                    </div>
                    <span className="text-[11px] text-on-surface-variant/50">{shapData.length} variables · pág. {shapPage + 1}/{totalPages}</span>
                    <div className="flex items-center gap-sm">
                      <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Acrecentamiento del Riesgo (SHAP &gt; 0)</span>
                      <div className="w-3 h-3 rounded-full bg-orange-500" />
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* ═══ TAB CONTENT: LOCAL SHAP ═══ */}
        {activeSubtab === "Local SHAP" && (
          <div className="max-w-4xl mx-auto w-full flex flex-col gap-lg animate-fade-in">

            {/* ── Historial de simulaciones ── */}
            <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] overflow-hidden">
              <div className="flex items-center justify-between px-lg py-md border-b border-outline-variant">
                <div>
                  <h3 className="text-headline-md text-on-surface font-bold">Historial de Simulaciones</h3>
                  <p className="text-label-md text-on-surface-variant mt-xs">
                    Selecciona una simulación para explicar con SHAP
                  </p>
                </div>
                {simulationHistory.length > 0 && (
                  <button
                    onClick={onClearHistory}
                    className="text-label-md text-error hover:underline flex items-center gap-xs cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[16px]">delete_sweep</span> Limpiar
                  </button>
                )}
              </div>

              {simulationHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-xl text-on-surface-variant gap-md">
                  <span className="material-symbols-outlined text-[40px] opacity-30">sensors</span>
                  <p className="text-label-md text-center">
                    Primero ejecuta una simulación en el <strong className="text-primary">Predictor</strong> y vuelve aquí para explicarla.
                  </p>
                </div>
              ) : (
                <ul className="divide-y divide-outline-variant max-h-64 overflow-y-auto">
                  {simulationHistory.map((sim, idx) => {
                    const isSelected = idx === selectedIdx;
                    const timeLabel = sim.timestamp
                      ? sim.timestamp.toLocaleTimeString("es-PE", { hour: "2-digit", minute: "2-digit" })
                      : "";
                    return (
                      <li
                        key={sim.id}
                        onClick={() => setSelectedIdx(idx)}
                        className={`flex items-center gap-md px-lg py-sm cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-primary/8 border-l-4 border-l-primary"
                            : "hover:bg-surface-container-low border-l-4 border-l-transparent"
                        }`}
                      >
                        <span className={`material-symbols-outlined text-[20px] ${isSelected ? "text-primary" : "text-on-surface-variant"}`}>
                          {isSelected ? "radio_button_checked" : "radio_button_unchecked"}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className={`text-label-md font-bold truncate ${isSelected ? "text-primary" : "text-on-surface"}`}>
                            {sim.adm_1_name} <span className="font-normal text-on-surface-variant">({sim.country})</span>
                          </p>
                          <p className="text-[12px] text-on-surface-variant">
                            Mes: <strong>{MONTH_NAMES[(sim.mes ?? 1) - 1]}</strong> · {Object.keys(sim.clima_overrides ?? {}).length} variables
                          </p>
                        </div>
                        <div className="flex flex-col items-end gap-xs flex-shrink-0">
                          {idx === 0 && (
                            <span className="px-xs py-[2px] bg-primary/10 text-primary text-[10px] font-bold rounded-full">ÚLTIMA</span>
                          )}
                          <span className="text-[11px] text-on-surface-variant">{timeLabel}</span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            {/* ── Panel de explicación ── */}
            {lastSimulation && (
              <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] flex flex-col">
                <div className="mb-lg">
                  <h3 className="text-headline-md text-on-surface font-bold">SHAP Local — Explicación de Simulación</h3>
                  <p className="text-label-md text-on-surface-variant mt-xs">
                    Descompone qué variables contribuyeron más a la predicción y en qué dirección.
                  </p>
                </div>

                {/* Contexto + botón */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-md mb-lg p-md rounded-xl border border-outline-variant bg-surface-container-low">
                  <div className="flex-1 space-y-xs">
                    <p className="text-label-md font-bold text-on-surface">
                      {lastSimulation.country} · {lastSimulation.adm_1_name}
                    </p>
                    <p className="text-[12px] text-on-surface-variant">
                      Mes objetivo: <strong>{MONTH_NAMES[(lastSimulation.mes ?? 1) - 1]}</strong> · {Object.keys(lastSimulation.clima_overrides ?? {}).length} variables configuradas
                    </p>
                  </div>
                  <button
                    onClick={handleAnalyzeLocal}
                    disabled={localLoading}
                    className="px-lg py-sm bg-primary text-on-primary rounded-lg text-label-md font-bold hover:bg-primary/90 transition-colors flex items-center gap-sm cursor-pointer disabled:opacity-55 whitespace-nowrap"
                  >
                    {localLoading
                      ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span> Analizando...</>
                      : <><span className="material-symbols-outlined text-[16px]">analytics</span> Explicar simulación</>}
                  </button>
                </div>

                {/* Error */}
                {localError && (
                  <div className="bg-error-container p-md rounded-lg flex items-center gap-md mb-md">
                    <span className="material-symbols-outlined text-on-error-container">error</span>
                    <p className="text-label-md text-on-error-container font-medium">{localError}</p>
                  </div>
                )}

                {/* Empty state */}
                {!localResult && !localLoading && !localError && (
                  <div className="flex flex-col items-center justify-center py-lg text-on-surface-variant gap-md">
                    <span className="material-symbols-outlined text-[48px] opacity-30">bar_chart_4_bars</span>
                    <p className="text-label-md">Haz clic en "Explicar simulación" para analizar</p>
                  </div>
                )}

                {/* Results */}
                {localResult && (
                  <div className="animate-fade-in">
                    <div className="flex items-center gap-lg mb-lg p-md rounded-lg border border-outline-variant bg-surface-container-low">
                      <div>
                        <p className="text-label-md text-on-surface-variant">Predicción Ensemble</p>
                        <p className="text-headline-lg font-bold text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {localResult.prediction?.toFixed(1)} <span className="text-label-md font-normal opacity-60">casos/100k</span>
                        </p>
                      </div>
                      <div
                        className="px-md py-xs rounded-full text-label-md font-bold text-white"
                        style={{ backgroundColor: localResult.riesgo?.color ?? "#10b981" }}
                      >
                        {localResult.riesgo?.nivel ?? "—"}
                      </div>
                    </div>

                    <p className="text-label-md font-bold text-on-surface-variant uppercase tracking-wider mb-md">
                      Top {localResult.shapArr.length} variables por impacto SHAP
                    </p>
                    <div className="space-y-md">
                      {(() => {
                        const maxAbs = Math.max(...localResult.shapArr.map((f) => Math.abs(f.value)), 0.001);
                        return localResult.shapArr.map((feat) => {
                          const pct = (Math.abs(feat.value) / maxAbs) * 100;
                          const isNeg = feat.value < 0;
                          return (
                            <div key={feat.feature} className="space-y-xs">
                              <div className="flex justify-between items-center" style={{ fontVariantNumeric: "tabular-nums" }}>
                                <span className="text-label-md text-on-surface font-medium">{feat.feature}</span>
                                <span className={`text-label-md font-bold ${isNeg ? "text-blue-600" : "text-orange-500"}`}>
                                  {feat.value > 0 ? "+" : ""}{feat.value.toFixed(4)}
                                </span>
                              </div>
                              <div className="w-full bg-surface-container-low h-3.5 rounded-full overflow-hidden">
                                <div
                                  className={`chart-bar h-full rounded-full bg-gradient-to-r ${isNeg ? "from-blue-600 to-blue-400" : "from-orange-400 to-orange-600"}`}
                                  style={{ width: `${Math.max(pct, 3)}%` }}
                                ></div>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>

                    <div className="mt-lg flex justify-between items-center border-t border-outline-variant pt-md">
                      <div className="flex items-center gap-sm">
                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                        <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Reduce riesgo (SHAP &lt; 0)</span>
                      </div>
                      <div className="flex items-center gap-sm">
                        <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Aumenta riesgo (SHAP &gt; 0)</span>
                        <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ═══ Bottom: Scientific Interpretation ═══ */}
      <div className="bg-[#eff6ff] dark:bg-sky-950/20 border border-outline-variant p-lg rounded-xl flex gap-lg items-start mb-lg max-w-4xl mx-auto w-full">
        <div className="p-sm bg-white dark:bg-zinc-800 rounded-lg shadow-sm flex-shrink-0">
          <span className="material-symbols-outlined text-primary text-[32px]">lightbulb</span>
        </div>
        <div>
          <h4 className="text-headline-md text-on-surface font-bold mb-sm">
            Interpretación Científica de Coeficientes Shapley
          </h4>
          <div className="space-y-md text-on-surface-variant text-body-md leading-relaxed">
            <p>
              Los valores SHAP (SHapley Additive exPlanations) descomponen la predicción final en la contribución
              individual de cada variable climática y epidemiológica. Para la toma de decisiones en salud pública,
              un valor positivo indica un <strong className="text-on-surface">aumento del riesgo relativo</strong>,
              mientras que un valor negativo sugiere <strong className="text-on-surface">atenuación de la incidencia</strong>.
            </p>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-md">
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>incidencia_lag1:</strong> El historial inmediato de casos sigue siendo el predictor más
                  robusto de brotes epidémicos.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>tmax_promedio:</strong> Las temperaturas máximas extremas aceleran el ciclo metabólico del
                  vector <em>Aedes aegypti</em>.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>precipitación:</strong> Los volúmenes de lluvia crean hábitats de criaderos acuáticos
                  propicios para el vector.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>humedad_relativa:</strong> La humedad elevada extiende la supervivencia del mosquito
                  adulto y su capacidad de vuelo.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
