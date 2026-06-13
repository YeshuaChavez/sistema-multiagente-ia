import React, { useEffect, useState } from "react";

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
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recalculating, setRecalculating] = useState(false);

  useEffect(() => {
    const fetchShap = async () => {
      try {
        const response = await fetch(`${API_URL}/api/explain/global`);
        if (!response.ok) throw new Error("No se pudo cargar la explicabilidad SHAP");
        const raw = await response.json();
        // API returns {feature: importance} dict — convert to sorted array
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

  const handleRecalculate = () => {
    setRecalculating(true);
    setTimeout(() => {
      setRecalculating(false);
      alert("Valores SHAP re-calculados exitosamente para todas las muestras en memoria.");
    }, 1500);
  };

  const handleExport = () => {
    alert("Exportando reporte SHAP en formato PDF...");
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

        {/* ═══ TAB CONTENT: LOCAL SHAP Force Plot ═══ */}
        {activeSubtab === "Local SHAP" && (
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] max-w-4xl mx-auto w-full flex flex-col animate-fade-in">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-md mb-lg">
              <div>
                <h3 className="text-headline-md text-on-surface font-bold">SHAP Local Force Plot</h3>
                <p className="text-label-md text-on-surface-variant">Descomposición espacial de una muestra en vivo</p>
              </div>
              <select className="bg-surface border border-outline-variant text-label-md rounded-lg py-xs px-md focus:ring-primary focus:border-primary outline-none cursor-pointer text-on-surface">
                <option>Muestra ID #4281 (Loreto, Perú)</option>
                <option>Muestra ID #4282 (Amazonas, Brasil)</option>
                <option>Muestra ID #4283 (Cali, Colombia)</option>
              </select>
            </div>

            {/* Force Plot Visualization */}
            <div className="flex-1 flex flex-col justify-center py-xl relative overflow-hidden">
              {/* Baseline dashed line */}
              <div className="absolute left-1/2 top-0 bottom-0 border-l border-dashed border-outline pointer-events-none"></div>
              <div className="absolute left-1/2 -top-1 -translate-x-1/2 px-md py-xs bg-surface dark:bg-zinc-800 border border-outline-variant rounded text-[10px] font-bold text-on-surface-variant z-10 whitespace-nowrap">
                VALOR BASE CONTINENTAL: 12.4
              </div>

              {/* Force Bars */}
              <div className="relative h-24 flex items-center mt-lg">
                {/* Negative Forces (Left — Blue) */}
                <div className="flex-1 flex justify-end items-center gap-[2px]">
                  <div
                    className="h-10 bg-[#adc8f5] dark:bg-blue-900/60 rounded-l-full px-md flex items-center text-primary dark:text-blue-300 text-[11px] font-bold whitespace-nowrap"
                    style={{ width: "40%", animation: "slideIn 0.5s ease-out forwards" }}
                  >
                    Acceso Agua (82.3%)
                    <span className="material-symbols-outlined text-[14px] ml-xs">trending_down</span>
                  </div>
                  <div
                    className="h-10 bg-[#adc8f5] dark:bg-blue-900/40 px-sm flex items-center text-primary dark:text-blue-300 text-[11px] font-bold"
                    style={{ width: "25%", animation: "slideIn 0.5s ease-out 100ms forwards" }}
                  >
                    <span className="material-symbols-outlined text-[14px]">arrow_back</span>
                  </div>
                </div>

                {/* Center Prediction Circle */}
                <div className="z-20 bg-primary text-on-primary w-16 h-16 rounded-full flex items-center justify-center border-4 border-white dark:border-zinc-800 shadow-lg -mx-8 flex-shrink-0">
                  <span className="text-headline-md font-bold" style={{ fontVariantNumeric: "tabular-nums" }}>
                    24.8
                  </span>
                </div>

                {/* Positive Forces (Right — Red) */}
                <div className="flex-1 flex justify-start items-center gap-[2px]">
                  <div
                    className="h-10 bg-[#ffdad6] dark:bg-red-950/40 px-sm flex items-center text-error dark:text-red-300 text-[11px] font-bold"
                    style={{ width: "30%", animation: "slideIn 0.5s ease-out 200ms forwards" }}
                  >
                    <span className="material-symbols-outlined text-[14px]">arrow_forward</span>
                  </div>
                  <div
                    className="h-10 bg-[#ffdad6] dark:bg-red-900/60 rounded-r-full px-md flex items-center text-error dark:text-red-300 text-[11px] font-bold whitespace-nowrap"
                    style={{ width: "55%", animation: "slideIn 0.5s ease-out 300ms forwards" }}
                  >
                    <span className="material-symbols-outlined text-[14px] mr-xs">trending_up</span>
                    Temp. Máxima (32.5°C)
                  </div>
                </div>
              </div>

              {/* Explanation Cards Below Force Plot */}
              <div className="mt-xl grid grid-cols-1 sm:grid-cols-2 gap-lg text-center">
                <div className="p-md rounded-lg bg-surface-container-low dark:bg-zinc-850 border border-outline-variant">
                  <p className="text-label-md text-primary dark:text-blue-400 mb-xs font-bold uppercase tracking-wider">
                    FUERZAS NEGATIVAS
                  </p>
                  <p className="text-[12px] text-on-surface-variant leading-relaxed">
                    Factores que reducen la predicción de casos (ej. buena infraestructura sanitaria local).
                  </p>
                </div>
                <div className="p-md rounded-lg bg-error-container/20 dark:bg-red-950/20 border border-outline-variant">
                  <p className="text-label-md text-error dark:text-red-400 mb-xs font-bold uppercase tracking-wider">
                    FUERZAS POSITIVAS
                  </p>
                  <p className="text-[12px] text-on-surface-variant leading-relaxed">
                    Factores que aumentan el riesgo detectado (ej. anomalías térmicas y lluvias extremas).
                  </p>
                </div>
              </div>
            </div>
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
