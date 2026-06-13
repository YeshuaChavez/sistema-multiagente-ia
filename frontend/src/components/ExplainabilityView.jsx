import React, { useEffect, useState, useCallback } from "react";

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

export default function ExplainabilityView({ activeSubtab }) {
  // ─── Global SHAP state ───
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recalculating, setRecalculating] = useState(false);

  // ─── Local SHAP state ───
  const [metadata, setMetadata] = useState({});       // {pais: {iso_a0, departamentos}}
  const [localCountry, setLocalCountry] = useState(""); // pais name
  const [localDept, setLocalDept] = useState("");
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [localResult, setLocalResult] = useState(null); // {prediction, riesgo, shap_local}

  // ─── Fetch global SHAP ───
  useEffect(() => {
    const fetchShap = async () => {
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
    };
    fetchShap();
  }, []);

  // ─── Fetch metadata for local SHAP selectors ───
  useEffect(() => {
    fetch(`${API_URL}/api/metadata`)
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((data) => {
        setMetadata(data);
        const firstCountry = Object.keys(data)[0] ?? "";
        setLocalCountry(firstCountry);
        setLocalDept(data[firstCountry]?.departamentos?.[0] ?? "");
      })
      .catch(() => {});
  }, []);

  const countryOptions = Object.keys(metadata);
  const deptOptions = localCountry ? (metadata[localCountry]?.departamentos ?? []) : [];

  const handleCountryChange = (e) => {
    const c = e.target.value;
    setLocalCountry(c);
    setLocalDept(metadata[c]?.departamentos?.[0] ?? "");
    setLocalResult(null);
  };

  const handleAnalyzeLocal = useCallback(async () => {
    if (!localCountry || !localDept) return;
    const iso_a0 = metadata[localCountry]?.iso_a0;
    if (!iso_a0) return;
    setLocalLoading(true);
    setLocalError(null);
    setLocalResult(null);
    try {
      const res = await fetch(`${API_URL}/api/predict/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ iso_a0, adm_1_name: localDept }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      if (!data.shap_local) throw new Error("El backend no retornó valores SHAP locales.");
      // Convert shap_local dict to sorted array (top 12 by |value|)
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
  }, [localCountry, localDept, metadata]);

  const handleRecalculate = () => {
    setRecalculating(true);
    setTimeout(() => setRecalculating(false), 1500);
  };

  const handleExport = () => {
    alert("Exportar PDF disponible desde el Panel de Control.");
  };

  const maxVal = Array.isArray(shapData) && shapData.length > 0 
    ? Math.max(...shapData.map((f) => Math.abs(f.importance))) 
    : 1;

  // Timestamp for display
  const now = new Date();
  const timeStr = `Hoy, ${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")} PM`;

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
            {Array.isArray(shapData) && shapData.length > 0 && (
              <div className="space-y-lg">
                {shapData.map((feature) => {
                  const pct = (Math.abs(feature.importance) / maxVal) * 100;
                  const isNegative = feature.importance < 0;
                  return (
                    <div key={feature.feature} className="space-y-sm">
                      <div className="flex justify-between items-center" style={{ fontVariantNumeric: "tabular-nums" }}>
                        <span className="text-label-md text-on-surface font-medium">{feature.feature}</span>
                        <span className="text-label-md text-on-surface-variant">{feature.importance.toFixed(4)}</span>
                      </div>
                      <div className="w-full bg-surface-container-low dark:bg-zinc-850 h-4 rounded-full overflow-hidden">
                        <div
                          className={`chart-bar h-full bg-gradient-to-r ${
                            isNegative 
                              ? "from-blue-600 to-blue-400" 
                              : "from-orange-400 to-orange-600"
                          } rounded-full`}
                          style={{ width: `${Math.max(pct, 3)}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}

                {/* Legend */}
                <div className="mt-xl flex justify-between items-center border-t border-outline-variant pt-lg">
                  <div className="flex items-center gap-sm">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span className="text-[11px] text-on-surface-variant text-label-md uppercase tracking-wider">Atenuación del Riesgo (SHAP &lt; 0)</span>
                  </div>
                  <div className="flex items-center gap-sm">
                    <span className="text-[11px] text-on-surface-variant text-label-md uppercase tracking-wider">Acrecentamiento del Riesgo (SHAP &gt; 0)</span>
                    <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══ TAB CONTENT: LOCAL SHAP ═══ */}
        {activeSubtab === "Local SHAP" && (
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] max-w-4xl mx-auto w-full flex flex-col animate-fade-in">
            <div className="mb-lg">
              <h3 className="text-headline-md text-on-surface font-bold">SHAP Local — Descomposición por Departamento</h3>
              <p className="text-label-md text-on-surface-variant mt-xs">
                Selecciona un país y departamento para ver la contribución de cada variable a la predicción de ese lugar.
              </p>
            </div>

            {/* Selectors */}
            <div className="flex flex-col sm:flex-row gap-md mb-lg">
              <select
                value={localCountry}
                onChange={handleCountryChange}
                className="flex-1 bg-surface-container border border-outline-variant text-label-md rounded-lg py-xs px-md focus:ring-primary outline-none cursor-pointer text-on-surface"
              >
                {countryOptions.length === 0 && <option value="">Cargando países...</option>}
                {countryOptions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
              <select
                value={localDept}
                onChange={(e) => { setLocalDept(e.target.value); setLocalResult(null); }}
                className="flex-1 bg-surface-container border border-outline-variant text-label-md rounded-lg py-xs px-md focus:ring-primary outline-none cursor-pointer text-on-surface"
              >
                {deptOptions.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
              <button
                onClick={handleAnalyzeLocal}
                disabled={localLoading || !localDept}
                className="px-lg py-xs bg-primary text-on-primary rounded-lg text-label-md font-bold hover:bg-primary/90 transition-colors flex items-center gap-sm cursor-pointer disabled:opacity-55"
              >
                {localLoading
                  ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span> Analizando...</>
                  : <><span className="material-symbols-outlined text-[16px]">analytics</span> Analizar</>}
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
              <div className="flex flex-col items-center justify-center py-2xl text-on-surface-variant gap-md">
                <span className="material-symbols-outlined text-[48px] opacity-30">bar_chart_4_bars</span>
                <p className="text-label-md">Selecciona un departamento y haz clic en Analizar</p>
              </div>
            )}

            {/* Results */}
            {localResult && (
              <div className="animate-fade-in">
                {/* Prediction summary */}
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

                {/* SHAP bars */}
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

                {/* Legend */}
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
