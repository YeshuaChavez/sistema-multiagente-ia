import React, { useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Risk badge styling
const riskStyles = {
  "Endémico": { bg: "bg-[#10b981]/10", text: "text-[#00714d]", border: "border-[#10b981]/30", label: "Endémico", icon: "check_circle", ensemble: "bg-[#10b981]", glow: "badge-glow-green" },
  Alerta:     { bg: "bg-[#ea580c]/10", text: "text-[#ea580c]", border: "border-[#ea580c]/30", label: "Alerta",    icon: "warning",       ensemble: "bg-[#ea580c]", glow: "badge-glow-orange" },
  Epidemia:   { bg: "bg-[#ba1a1a]/10", text: "text-[#ba1a1a]", border: "border-[#ba1a1a]/30", label: "Epidemia",  icon: "emergency",     ensemble: "bg-[#ba1a1a]", glow: "badge-glow-red" },
};

// Configuration of ranges, labels, and steps for model features
// Ordered by SHAP global importance — principal = top drivers shown by default
const FEATURE_DEFS = {
  // ── Principal: ordered by SHAP importance ──
  incidencia_lag1:  { label: "Incidencia Dengue Mes Anterior (t-1)", min: 0,   max: 500, step: 0.1, unit: " /100k", section: "principal" },
  tmax_promedio:    { label: "Temperatura Máxima Promedio (°C)",      min: 15,  max: 45,  step: 0.1, unit: "°C",     section: "principal" },
  precipitacion:    { label: "Precipitación Acumulada (mm)",           min: 0,   max: 500, step: 0.5, unit: " mm",    section: "principal" },
  tmin_promedio:    { label: "Temperatura Mínima Promedio (°C)",       min: 10,  max: 30,  step: 0.1, unit: "°C",     section: "principal" },
  humedad_promedio: { label: "Humedad Relativa Promedio (%)",           min: 0,   max: 100, step: 1,   unit: "%",      section: "principal" },

  // ── Advanced: lags autorregresivos (mayor SHAP primero) ──
  incidencia_lag2:       { label: "Incidencia Dengue Lag 2 (t-2)",    min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_lag3:       { label: "Incidencia Dengue Lag 3 (t-3)",    min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_lag4:       { label: "Incidencia Dengue Lag 4 (t-4)",    min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_lag5:       { label: "Incidencia Dengue Lag 5 (t-5)",    min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_lag6:       { label: "Incidencia Dengue Lag 6 (t-6)",    min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  // Vecinos espaciales
  incidencia_vecinos_lag1: { label: "Incidencia Vecina Lag 1 (t-1)", min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_vecinos_lag2: { label: "Incidencia Vecina Lag 2 (t-2)", min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  incidencia_vecinos_lag3: { label: "Incidencia Vecina Lag 3 (t-3)", min: 0,  max: 500, step: 0.1, unit: " /100k",   section: "advanced" },
  // Lags climáticos
  precipitacion_lag1: { label: "Precipitación Lag 1 (t-1)", min: 0,  max: 500, step: 0.5, unit: " mm", section: "advanced" },
  precipitacion_lag2: { label: "Precipitación Lag 2 (t-2)", min: 0,  max: 500, step: 0.5, unit: " mm", section: "advanced" },
  precipitacion_lag3: { label: "Precipitación Lag 3 (t-3)", min: 0,  max: 500, step: 0.5, unit: " mm", section: "advanced" },
  tmax_lag1: { label: "Temp. Máxima Lag 1 (t-1)", min: 15, max: 45, step: 0.1, unit: "°C", section: "advanced" },
  tmax_lag2: { label: "Temp. Máxima Lag 2 (t-2)", min: 15, max: 45, step: 0.1, unit: "°C", section: "advanced" },
  tmax_lag3: { label: "Temp. Máxima Lag 3 (t-3)", min: 15, max: 45, step: 0.1, unit: "°C", section: "advanced" },
  tmin_lag1: { label: "Temp. Mínima Lag 1 (t-1)", min: 10, max: 30, step: 0.1, unit: "°C", section: "advanced" },
  tmin_lag2: { label: "Temp. Mínima Lag 2 (t-2)", min: 10, max: 30, step: 0.1, unit: "°C", section: "advanced" },
  tmin_lag3: { label: "Temp. Mínima Lag 3 (t-3)", min: 10, max: 30, step: 0.1, unit: "°C", section: "advanced" },
  humedad_lag1: { label: "Humedad Lag 1 (t-1)", min: 0, max: 100, step: 1, unit: "%", section: "advanced" },
  humedad_lag2: { label: "Humedad Lag 2 (t-2)", min: 0, max: 100, step: 1, unit: "%", section: "advanced" },
  humedad_lag3: { label: "Humedad Lag 3 (t-3)", min: 0, max: 100, step: 1, unit: "%", section: "advanced" },
  // Socioeconómicas — al final por correlación confundida
  densidad_poblacion: { label: "Densidad de Población (hab/km²)",  min: 0, max: 1000, step: 1,   unit: " hab/km²", section: "advanced" },
  agua_basica:        { label: "Acceso Agua Potable Básica (%)",   min: 0, max: 100,  step: 0.1, unit: "%",        section: "advanced" },
};

// Generates a mock history array for local demo fallback
const generateMockHistory = (country, dept) => {
  const records = [];
  const startYear = 2014;
  const currentDate = new Date();
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1;
  const hash = (dept || "").length + 7;
  let baseCases = 40 + (hash % 6) * 15;
  
  for (let year = startYear; year <= currentYear; year++) {
    for (let month = 1; month <= 12; month++) {
      if (year === currentYear && month > currentMonth) break;
      
      const tmax = 28 + Math.sin(month / 2) * 4 + (year % 3) * 0.5;
      const tmin = 18 + Math.sin(month / 2) * 3 + (year % 2) * 0.3;
      const precip = 80 + Math.cos(month / 1.5) * 60 + (year % 4) * 15;
      const humedad = 70 + Math.sin(month / 3) * 15;
      
      const seasonFactor = Math.sin((month - 2) * Math.PI / 6) + 1; // 0 to 2
      const yearFactor = 0.6 + (Math.sin(year) + 1) * 0.6;
      const cases = Math.max(5, Math.floor(baseCases * seasonFactor * yearFactor + (hash % 3) * 8));
      const pop = 400000 + (hash % 5) * 100000;
      const incidence = (cases / pop) * 100000;
      
      records.push({
        fecha: `${year}-${month.toString().padStart(2, "0")}`,
        ano: year,
        mes: month,
        casos: cases,
        incidencia: parseFloat(incidence.toFixed(2)),
        tmax: parseFloat(tmax.toFixed(1)),
        tmin: parseFloat(tmin.toFixed(1)),
        precipitacion: parseFloat(precip.toFixed(1)),
        humedad: parseFloat(humedad.toFixed(0))
      });
    }
  }
  return records;
};

// Frontend calculation of mock prediction based on feature values
const runMockPrediction = (values, country, dept) => {
  const tmax = values.tmax_promedio ?? 30.5;
  const precip = values.precipitacion ?? 120.0;
  const lag1 = values.incidencia_lag1 ?? 35.0;
  const agua = values.agua_basica ?? 82.5;
  
  let baseInc = 15.0;
  if (tmax > 32) baseInc += (tmax - 32) * 6;
  if (tmax < 24) baseInc -= (24 - tmax) * 2;
  if (precip > 150) baseInc += (precip - 150) * 0.15;
  baseInc += lag1 * 0.75;
  if (agua < 90) baseInc += (90 - agua) * 0.5;
  
  baseInc = Math.max(1.0, baseInc);
  
  const pred_ml = baseInc * (0.94 + Math.sin(tmax) * 0.04);
  const pred_lstm = baseInc * (1.0 + Math.sin(tmax + precip * 0.01) * 0.03);
  const pred_ens = (pred_ml + pred_lstm) / 2.0;

  const isLowRiskDept = dept && dept.toUpperCase() === "AGUASCALIENTES";
  const p50 = isLowRiskDept ? 0.05 : 2.8;
  const p90 = isLowRiskDept ? 0.8 : 64.0;

  const getMockRisk = (val) => {
    if (val <= p50) return { nivel: "Endémico", codigo: "endemico", color: "#10b981" };
    if (val <= p90) return { nivel: "Alerta",   codigo: "alerta",   color: "#f97316" };
    return { nivel: "Epidemia", codigo: "epidemia", color: "#ef4444" };
  };

  return {
    prediccion_ml: pred_ml,
    riesgo_ml: getMockRisk(pred_ml),
    prediccion_lstm: pred_lstm,
    riesgo_lstm: getMockRisk(pred_lstm),
    prediccion_ensemble: pred_ens,
    riesgo_ensemble: getMockRisk(pred_ens),
    features_usadas: values,
    percentiles_locales: { p25, p50, p90 }
  };
};

export default function PredictorView({
  metadata,
  selectedCountry,
  selectedDept,
  setSelectedCountry,
  setSelectedDept,
  activeSubtab,
  backendStatus,
  onSimulationComplete,
}) {
  const [sliderValues, setSliderValues] = useState({});
  const [enso, setEnso] = useState("neutral"); // "neutral" | "nino" | "nina"
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedPage, setAdvancedPage] = useState(0);
  const ADV_PAGE_SIZE = 7;
  const [targetMes, setTargetMes] = useState(new Date().getMonth() + 1);

  const currentDate = new Date();
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1;
  const monthNames = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"];
  const currentMonthName = monthNames[currentMonth - 1];
  
  // Predictor States
  const [loading, setLoading] = useState(false);
  const [agentPhase, setAgentPhase] = useState(0);
  const [loadingBaseline, setLoadingBaseline] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  // Mobile-only: alterna entre panel de configuración y panel de resultado
  const [mobileView, setMobileView] = useState("config");

  // History States
  const [historicalData, setHistoricalData] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const countries = metadata ? Object.keys(metadata) : [];
  const departments = metadata && selectedCountry ? metadata[selectedCountry]?.departamentos || [] : [];

  // Auto-select department if none selected
  useEffect(() => {
    if (departments.length > 0 && !departments.includes(selectedDept)) {
      setSelectedDept(departments[0]);
    }
  }, [selectedCountry, departments, selectedDept, setSelectedDept]);

  // Volver a "config" cuando el usuario cambia de ubicación
  useEffect(() => { setMobileView("config"); }, [selectedCountry, selectedDept]);

  // Load baseline values when selected location changes
  useEffect(() => {
    if (!selectedCountry || !selectedDept) return;
    
    const fetchBaseline = async () => {
      setLoadingBaseline(true);
      setError(null);
      try {
        const isoCode = metadata[selectedCountry]?.iso_a0 || selectedCountry;
        
        if (backendStatus === "offline") {
          // Initialize mock baseline values
          const defaults = {};
          Object.keys(FEATURE_DEFS).forEach(k => {
            defaults[k] = (FEATURE_DEFS[k].min + FEATURE_DEFS[k].max) / 2;
          });
          defaults.agua_basica = 82.5;
          defaults.tmax_promedio = 32.5;
          defaults.tmin_promedio = 22.0;
          defaults.humedad_promedio = 78.0;
          defaults.precipitacion = 120.4;
          defaults.incidencia_lag1 = 45.0;
          defaults.densidad_poblacion = 150.0;
          setSliderValues(defaults);
          setResult(null);
        } else {
          // Endpoint ligero: solo devuelve features sin correr ningún modelo
          const params = new URLSearchParams({ iso_a0: isoCode, adm_1_name: selectedDept });
          const res = await fetch(`${API_URL}/api/features?${params}`);
          if (res.ok) {
            const data = await res.json();
            if (data.features) {
              setSliderValues(data.features);
            }
            setResult(null);
          } else {
            console.warn("Fallo en /api/features, usando valores por defecto...");
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingBaseline(false);
      }
    };

    fetchBaseline();
  }, [selectedCountry, selectedDept, backendStatus, metadata]);

  // Load historical records when Histórico subtab is active
  useEffect(() => {
    if (activeSubtab === "Histórico" && selectedCountry && selectedDept) {
      const loadHistory = async () => {
        setLoadingHistory(true);
        try {
          const isoCode = metadata[selectedCountry]?.iso_a0 || selectedCountry;
          if (backendStatus === "offline") {
            setHistoricalData(generateMockHistory(selectedCountry, selectedDept));
          } else {
            const res = await fetch(`${API_URL}/api/historical?iso_a0=${isoCode}&adm_1_name=${selectedDept}`);
            if (res.ok) {
              const data = await res.json();
              setHistoricalData(data);
            } else {
              setHistoricalData(generateMockHistory(selectedCountry, selectedDept));
            }
          }
        } catch (err) {
          console.warn("Fallo carga de histórico real, cargando simulado...", err);
          setHistoricalData(generateMockHistory(selectedCountry, selectedDept));
        } finally {
          setLoadingHistory(false);
        }
      };
      loadHistory();
    }
  }, [activeSubtab, selectedCountry, selectedDept, backendStatus, metadata]);

  const handleReset = () => {
    const defaults = {};
    Object.keys(FEATURE_DEFS).forEach(k => {
      defaults[k] = (FEATURE_DEFS[k].min + FEATURE_DEFS[k].max) / 2;
    });
    defaults.agua_basica = 82.5;
    defaults.tmax_promedio = 32.5;
    defaults.tmin_promedio = 22.0;
    defaults.humedad_promedio = 78.0;
    defaults.precipitacion = 120.4;
    defaults.incidencia_lag1 = 45.0;
    defaults.densidad_poblacion = 150.0;
    setSliderValues(defaults);
    setResult(null);
    setError(null);
  };

  const handlePredict = useCallback(async () => {
    if (!selectedCountry || !selectedDept) {
      setError("Selecciona un país y departamento antes de ejecutar la simulación.");
      return;
    }
    setLoading(true);
    setAgentPhase(1);
    setError(null);

    const startTime = Date.now();
    let apiResult = null;
    let apiError = null;

    // Iniciar temporizadores para las fases visuales de los agentes
    setTimeout(() => setAgentPhase(2), 450);
    setTimeout(() => setAgentPhase(3), 900);
    setTimeout(() => setAgentPhase(4), 1350);

    const isoCode = metadata[selectedCountry]?.iso_a0 || selectedCountry;

    const runPrediction = async () => {
      try {
        if (backendStatus === "offline") {
          // Fallback offline: Modelo matemático de simulación local
          apiResult = runMockPrediction(sliderValues, selectedCountry, selectedDept);
        } else {
          // Excluir features que siempre se recalculan en el backend
          const { mes_sin, mes_cos, incidencia_roll3, incidencia_roll6, ...cleanOverrides } = sliderValues;
          const body = {
            iso_a0: isoCode,
            adm_1_name: selectedDept,
            mes: targetMes,
            clima_overrides: {
              ...cleanOverrides,
              indicador_nino: enso === "nino" ? 1 : 0,
              indicador_nina: enso === "nina" ? 1 : 0,
            },
          };
          const res = await fetch(`${API_URL}/api/predict/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (!res.ok) {
            const d = await res.json();
            throw new Error(d.detail || "Error en simulación");
          }
          apiResult = await res.json();
        }
      } catch (err) {
        apiError = err.message;
      } finally {
        const elapsedTime = Date.now() - startTime;
        const minDuration = 1800; // 1.8 segundos de animación para la consola de consenso
        const remainingTime = Math.max(0, minDuration - elapsedTime);
        
        setTimeout(() => {
          if (apiError) {
            setError(apiError);
          } else {
            setResult(apiResult);
            if (window.innerWidth < 1024) setMobileView("resultado");
            if (onSimulationComplete && apiResult) {
              onSimulationComplete({
                iso_a0: isoCode,
                adm_1_name: selectedDept,
                country: selectedCountry,
                mes: targetMes,
                clima_overrides: sliderValues,
              });
            }
          }
          setLoading(false);
          setAgentPhase(0);
        }, remainingTime);
      }
    };

    runPrediction();
  }, [selectedCountry, selectedDept, sliderValues, targetMes, backendStatus, metadata, onSimulationComplete]);

  const handleSliderChange = (key, val) => {
    setSliderValues(prev => ({
      ...prev,
      [key]: val
    }));
  };

  const getRisk = (r) => riskStyles[r?.nivel] || riskStyles["Endémico"];


  // Filter keys by section
  const principalKeys = Object.keys(FEATURE_DEFS).filter(k => FEATURE_DEFS[k].section === "principal");
  const advancedKeys = Object.keys(FEATURE_DEFS).filter(k => FEATURE_DEFS[k].section === "advanced");

  // Local department percentiles (falling back to global ones if not simulated yet)
  const localP50 = result?.percentiles_locales?.p50 ?? 2.8;
  const localP90 = result?.percentiles_locales?.p90 ?? 64.0;

  return (
    <div className="w-full max-w-[1440px] mx-auto space-y-lg">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-sm mb-xs">
          <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>sensors</span>
          <span className="text-secondary font-label-md font-bold uppercase tracking-wider">Inferencia y Alerta Temprana</span>
        </div>
        <h2 className="text-headline-lg text-primary font-bold">Consola Predictiva — DenguePredict</h2>
        <p className="hidden sm:block text-on-surface-variant text-body-md mt-xs max-w-3xl">
          Configure las variables geoclimáticas y epidemiológicas del departamento de interés para obtener una estimación de riesgo de brote. Alterne entre simulación de variables, el histórico temporal y pautas científicas de alertas preventivas.
        </p>
      </div>

      {/* RENDER ACTIVE SUBTAB CONTENT */}

      {activeSubtab === "Simulación" && (
        <div className="flex flex-col gap-lg animate-fade-in">

          {/* ── Tab switcher: solo visible en móvil ── */}
          <div className="flex lg:hidden rounded-xl overflow-hidden border border-outline-variant">
            <button
              onClick={() => setMobileView("config")}
              className={`flex-1 py-sm text-label-md font-bold transition-colors flex items-center justify-center gap-xs cursor-pointer
                ${mobileView === "config" ? "bg-primary text-on-primary" : "bg-surface-container text-on-surface-variant hover:bg-surface-container-high"}`}
            >
              <span className="material-symbols-outlined text-[16px]">tune</span>
              Configurar
            </button>
            <button
              onClick={() => setMobileView("resultado")}
              className={`flex-1 py-sm text-label-md font-bold transition-colors flex items-center justify-center gap-xs cursor-pointer
                ${mobileView === "resultado" ? "bg-primary text-on-primary" : "bg-surface-container text-on-surface-variant hover:bg-surface-container-high"}`}
            >
              <span className="material-symbols-outlined text-[16px]">hub</span>
              Resultado
              {result && <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />}
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-lg">
          {/* ═══════════ LEFT COLUMN: PARAMETERS ═══════════ */}
          <section className={`lg:col-span-5 flex flex-col gap-lg ${mobileView === "config" ? "" : "hidden lg:flex"}`}>
            <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-lg flex flex-col gap-lg shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
              
              {/* Panel Header */}
              <div className="flex items-center justify-between">
                <h3 className="text-headline-md text-primary font-bold">Simulador de Variables</h3>
                <button
                  onClick={handleReset}
                  className="text-primary-container hover:bg-surface-container px-sm py-xs rounded-lg flex items-center gap-xs text-label-md transition-colors cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[18px]">restart_alt</span>
                  Reiniciar
                </button>
              </div>

              {/* ─── LOCALIZACIÓN ─── */}
              <div className="space-y-md">
                <div className="flex items-center gap-sm border-b border-outline-variant pb-xs">
                  <span className="material-symbols-outlined text-primary-fixed-dim">location_on</span>
                  <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">VIGILANCIA GEOGRÁFICA</h4>
                </div>
                <div className="grid grid-cols-2 gap-md pt-xs">
                  <div className="space-y-xs">
                    <label className="text-body-md text-on-surface-variant">País</label>
                    <select
                      value={selectedCountry}
                      onChange={(e) => {
                        const newCountry = e.target.value;
                        const firstDept = metadata?.[newCountry]?.departamentos?.[0] ?? "";
                        setSelectedCountry(newCountry);
                        setSelectedDept(firstDept);
                      }}
                      className="w-full bg-surface-container-high text-primary font-bold text-label-md px-sm py-2 rounded-lg border-none outline-none cursor-pointer"
                    >
                      <option value="">Seleccionar...</option>
                      {countries.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-xs">
                    <label className="text-body-md text-on-surface-variant">Departamento</label>
                    <select
                      value={selectedDept}
                      onChange={(e) => setSelectedDept(e.target.value)}
                      disabled={!selectedCountry}
                      className="w-full bg-surface-container-high text-primary font-bold text-label-md px-sm py-2 rounded-lg border-none outline-none cursor-pointer disabled:opacity-40"
                    >
                      <option value="">Seleccionar...</option>
                      {departments.map((d) => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="space-y-xs">
                  <label className="text-body-md text-on-surface-variant">Mes objetivo de predicción</label>
                  <select
                    value={targetMes}
                    onChange={(e) => setTargetMes(parseInt(e.target.value))}
                    className="w-full bg-surface-container-high text-primary font-bold text-label-md px-sm py-2 rounded-lg border-none outline-none cursor-pointer"
                  >
                    {monthNames.map((name, i) => (
                      <option key={i + 1} value={i + 1}>{name}</option>
                    ))}
                  </select>
                </div>
                {selectedCountry && selectedDept && (
                  <p className="text-[11px] text-on-surface-variant italic">
                    Pre-cargado con el último período histórico disponible del departamento. Ajusta los sliders para simular condiciones distintas.
                  </p>
                )}
              </div>

              {/* ─── VARIABLES PRINCIPALES ─── */}
              <div className="space-y-md">
                <div className="flex items-center gap-sm border-b border-outline-variant pb-xs">
                  <span className="material-symbols-outlined text-primary-fixed-dim">tune</span>
                  <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">VARIABLES PRINCIPALES</h4>
                </div>

                {/* ENSO toggle */}
                <div className="space-y-xs">
                  <label className="text-body-md text-on-surface-variant">Fenómeno ENSO</label>
                  <div className="flex rounded-xl overflow-hidden border border-outline-variant">
                    {[
                      { key: "neutral", label: "Neutro",   icon: "water",       color: "text-on-surface-variant" },
                      { key: "nino",    label: "El Niño",  icon: "wb_sunny",    color: "text-orange-500" },
                      { key: "nina",    label: "La Niña",  icon: "water_drop",  color: "text-sky-500" },
                    ].map((opt) => (
                      <button
                        key={opt.key}
                        type="button"
                        onClick={() => setEnso(opt.key)}
                        className={`flex-1 flex items-center justify-center gap-xs py-sm text-label-md font-bold transition-colors cursor-pointer
                          ${enso === opt.key
                            ? "bg-primary text-on-primary"
                            : "bg-surface-container text-on-surface-variant hover:bg-surface-container-high"
                          }`}
                      >
                        <span className={`material-symbols-outlined text-[15px] ${enso === opt.key ? "text-on-primary" : opt.color}`}
                          style={{ fontVariationSettings: "'FILL' 1" }}>
                          {opt.icon}
                        </span>
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-lg pt-xs">
                  {principalKeys.map((key) => {
                    const def = FEATURE_DEFS[key];
                    const val = sliderValues[key] ?? (def.min + def.max) / 2;
                    return (
                      <div key={key} className="space-y-xs">
                        <div className="flex justify-between items-center gap-sm">
                          <label className="text-body-md text-on-surface-variant flex-1 min-w-0">{def.label}</label>
                          <div className="flex items-center gap-xs flex-shrink-0">
                            <input
                              type="number"
                              min={def.min}
                              max={def.max}
                              step={def.step}
                              value={val}
                              onChange={(e) => { const n = parseFloat(e.target.value); if (!isNaN(n)) handleSliderChange(key, n); }}
                              onBlur={(e) => { const n = parseFloat(e.target.value); handleSliderChange(key, isNaN(n) ? def.min : Math.min(def.max, Math.max(def.min, n))); }}
                              className="w-20 text-right font-bold text-primary bg-surface-container-high px-xs py-0.5 rounded-lg text-label-md border border-outline-variant/40 focus:outline-none focus:ring-1 focus:ring-primary/50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                              style={{ fontVariantNumeric: "tabular-nums" }}
                            />
                            {def.unit && <span className="text-label-md text-on-surface-variant/60 w-6">{def.unit}</span>}
                          </div>
                        </div>
                        <input
                          type="range"
                          min={def.min}
                          max={def.max}
                          step={def.step}
                          value={val}
                          onChange={(e) => handleSliderChange(key, parseFloat(e.target.value))}
                          className="slider-custom"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* ─── COLLAPSIBLE ADVANCED LAGS ─── */}
              <div className="border-t border-outline-variant/50 pt-md">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full flex items-center justify-between text-label-md font-bold text-primary hover:bg-surface-container px-sm py-sm rounded-lg transition-colors cursor-pointer"
                >
                  <span className="flex items-center gap-sm">
                    <span className="material-symbols-outlined text-[18px]">history</span>
                    Rezagos Avanzados y Lags ({showAdvanced ? "Ocultar" : "Mostrar"} {advancedKeys.length} variables)
                  </span>
                  <span className="material-symbols-outlined text-[20px]">
                    {showAdvanced ? "expand_less" : "expand_more"}
                  </span>
                </button>

                {showAdvanced && (() => {
                  const totalPages = Math.ceil(advancedKeys.length / ADV_PAGE_SIZE);
                  const pageKeys   = advancedKeys.slice(advancedPage * ADV_PAGE_SIZE, (advancedPage + 1) * ADV_PAGE_SIZE);
                  return (
                    <div className="pt-md mt-sm border-t border-outline-variant/35 animate-fade-in">
                      <div className="grid grid-cols-1 gap-md">
                        {pageKeys.map((key) => {
                          const def = FEATURE_DEFS[key];
                          const val = sliderValues[key] ?? (def.min + def.max) / 2;
                          return (
                            <div key={key} className="space-y-xs">
                              <div className="flex justify-between items-center gap-xs">
                                <label className="text-[12px] text-on-surface-variant leading-tight flex-1 min-w-0">{def.label}</label>
                                <div className="flex items-center gap-xs flex-shrink-0">
                                  <input
                                    type="number"
                                    min={def.min}
                                    max={def.max}
                                    step={def.step}
                                    value={val}
                                    onChange={(e) => { const n = parseFloat(e.target.value); if (!isNaN(n)) handleSliderChange(key, n); }}
                                    onBlur={(e) => { const n = parseFloat(e.target.value); handleSliderChange(key, isNaN(n) ? def.min : Math.min(def.max, Math.max(def.min, n))); }}
                                    className="w-16 text-right font-bold text-primary/80 bg-surface-container px-xs py-0.5 rounded text-[11px] border border-outline-variant/30 focus:outline-none focus:ring-1 focus:ring-primary/40 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                    style={{ fontVariantNumeric: "tabular-nums" }}
                                  />
                                  {def.unit && <span className="text-[11px] text-on-surface-variant/50 w-5">{def.unit}</span>}
                                </div>
                              </div>
                              <input
                                type="range"
                                min={def.min}
                                max={def.max}
                                step={def.step}
                                value={val}
                                onChange={(e) => handleSliderChange(key, parseFloat(e.target.value))}
                                className="slider-custom h-1"
                              />
                            </div>
                          );
                        })}
                      </div>
                      {/* Paginación */}
                      <div className="flex items-center justify-between mt-md pt-sm border-t border-outline-variant/30">
                        <button
                          type="button"
                          disabled={advancedPage === 0}
                          onClick={() => setAdvancedPage(p => p - 1)}
                          className="flex items-center gap-xs text-[12px] text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors"
                        >
                          <span className="material-symbols-outlined text-[16px]">chevron_left</span>
                          Anterior
                        </button>
                        <span className="text-[11px] text-on-surface-variant">
                          Página {advancedPage + 1} / {totalPages}
                          <span className="ml-xs opacity-60">({advancedKeys.length} variables)</span>
                        </span>
                        <button
                          type="button"
                          disabled={advancedPage >= totalPages - 1}
                          onClick={() => setAdvancedPage(p => p + 1)}
                          className="flex items-center gap-xs text-[12px] text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors"
                        >
                          Siguiente
                          <span className="material-symbols-outlined text-[16px]">chevron_right</span>
                        </button>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* ─── ACCION PREDECIR ─── */}
              <button
                onClick={handlePredict}
                disabled={loading || !selectedCountry || !selectedDept}
                className="w-full bg-primary text-on-primary px-md py-3 rounded-xl text-label-md font-bold hover:opacity-90 transition-all flex items-center justify-center gap-sm cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <span className="material-symbols-outlined animate-spin text-[18px]">progress_activity</span>
                    Procesando inferencia multi-agente...
                  </>
                ) : loadingBaseline ? (
                  <>
                    <span className="material-symbols-outlined animate-spin text-[18px]">progress_activity</span>
                    Cargando datos del departamento...
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-[18px]">rocket_launch</span>
                    Ejecutar Inferencia de Alerta Temprana
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ═══════════ RIGHT COLUMN: RESULTS ═══════════ */}
          <section className={`lg:col-span-7 flex flex-col gap-lg ${mobileView === "resultado" ? "" : "hidden lg:flex"}`}>
            <div className="flex items-center justify-between">
              <h3 className="text-headline-md text-primary font-bold">Resultados de Proyección <span className="text-on-surface-variant font-normal text-body-md">(Tasas por 100k hab.)</span></h3>
              <div className="flex items-center gap-xs">
                <span className="w-3 h-3 bg-secondary rounded-full animate-pulse"></span>
                <span className="text-label-md text-secondary font-medium">Consenso IA</span>
              </div>
            </div>

            {error && (
              <div className="bg-error-container border border-outline-variant rounded-lg p-md flex items-center gap-md animate-shake">
                <span className="material-symbols-outlined text-on-error-container">error</span>
                <p className="text-label-md text-on-error-container font-medium">{error}</p>
              </div>
            )}

            {/* PLACEHOLDER */}
            {!result && !loading && !error && (
              <div className="flex flex-col gap-md">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
                  <div className="bg-white dark:bg-zinc-900 border-l-4 border-l-outline-variant border border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 opacity-50">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-label-md text-on-surface-variant">Agente ML · R²=91.5%</p>
                        <h4 className="text-headline-md text-primary font-bold">XGBoost</h4>
                      </div>
                      <span className="material-symbols-outlined text-outline">analytics</span>
                    </div>
                    <div className="flex items-end justify-between">
                      <span className="text-[48px] font-black text-outline" style={{ fontVariantNumeric: "tabular-nums" }}>—</span>
                      <span className="bg-surface-container text-on-surface-variant px-sm py-xs rounded-lg text-label-md">Esperando...</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-zinc-900 border-l-4 border-l-outline-variant border border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 opacity-50">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-label-md text-on-surface-variant">Agente Seq. · R²=90.4%</p>
                        <h4 className="text-headline-md text-primary font-bold">LSTM PyTorch</h4>
                      </div>
                      <span className="material-symbols-outlined text-outline">timeline</span>
                    </div>
                    <div className="flex items-end justify-between">
                      <span className="text-[48px] font-black text-outline" style={{ fontVariantNumeric: "tabular-nums" }}>—</span>
                      <span className="bg-surface-container text-on-surface-variant px-sm py-xs rounded-lg text-label-md">Esperando...</span>
                    </div>
                  </div>
                </div>
                <div className="bg-primary/5 border border-outline-variant rounded-xl p-xl flex flex-col items-center justify-center text-center opacity-50" style={{ minHeight: "280px" }}>
                  <div className="w-12 h-12 bg-surface-container rounded-full flex items-center justify-center mb-md">
                    <span className="material-symbols-outlined text-[32px] text-outline">hub</span>
                  </div>
                  <p className="text-label-md text-on-surface-variant uppercase tracking-widest font-bold">Consenso de Fusión</p>
                  <p className="text-headline-md text-on-surface-variant mt-sm">Configure variables geoclimáticas a la izquierda y presione ejecutar</p>
                </div>
              </div>
            )}

            {/* CONSOLA NEURAL DE CONSENSO DE AGENTES */}
            {loading && (
              <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-xl flex flex-col justify-center shadow-lg animate-fade-in" style={{ minHeight: "450px" }}>
                <div className="text-center mb-lg">
                  <div className="flex items-center justify-center gap-xs text-secondary mb-xs">
                    <span className="material-symbols-outlined animate-spin-slow text-[22px]">hub</span>
                    <span className="text-[12px] font-bold uppercase tracking-wider">Consenso Neural Activo</span>
                  </div>
                  <h4 className="text-headline-md text-primary font-bold">Procesando Inferencia Multi-Agente</h4>
                  <p className="text-body-md text-on-surface-variant max-w-md mx-auto mt-xs">
                    Coordinando agentes cognitivos de aprendizaje automático y deep learning en tiempo real.
                  </p>
                </div>

                {/* Agent Flow Timeline */}
                <div className="max-w-md mx-auto w-full space-y-md relative">
                  {/* Vertical connecting line */}
                  <div className="absolute left-[23px] top-6 bottom-6 w-[2px] bg-outline-variant/40 z-0" />

                  {[
                    {
                      phase: 1,
                      name: "Agente 1: Ingestor de Datos",
                      desc: "Recuperando datos históricos y clima de satélites NASA POWER",
                      icon: "cloud_download",
                    },
                    {
                      phase: 2,
                      name: "Agente 2: Procesador de Características",
                      desc: "Normalizando variables geográficas y estructurando desfases (Lags)",
                      icon: "tune",
                    },
                    {
                      phase: 3,
                      name: "Agente 3 & 4: Inferencia ML / DL",
                      desc: "Computación de XGBoost (Árboles) y LSTM PyTorch (Memoria Recurrente)",
                      icon: "memory",
                    },
                    {
                      phase: 4,
                      name: "Agente 5: Agente Consenso (Fusion)",
                      desc: "Fusión ponderada de predicciones y calibración de alerta local",
                      icon: "hub",
                    },
                  ].map((agent) => {
                    const isCompleted = agentPhase > agent.phase;
                    const isProcessing = agentPhase === agent.phase;
                    const isWaiting = agentPhase < agent.phase;

                    return (
                      <div
                        key={agent.phase}
                        className={`flex gap-md items-start p-sm rounded-xl transition-all duration-300 relative z-10 ${
                          isProcessing
                            ? "bg-primary/5 border border-primary/20 scale-[1.02] shadow-sm"
                            : "border border-transparent"
                        }`}
                      >
                        {/* Status Icon */}
                        <div
                          className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${
                            isCompleted
                              ? "bg-emerald-100 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400"
                              : isProcessing
                              ? "bg-primary text-on-primary shadow-lg animate-pulse"
                              : "bg-surface-container text-on-surface-variant/40"
                          }`}
                        >
                          {isCompleted ? (
                            <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                          ) : isProcessing ? (
                            <span className="material-symbols-outlined text-[20px] animate-spin">progress_activity</span>
                          ) : (
                            <span className="material-symbols-outlined text-[20px]">{agent.icon}</span>
                          )}
                        </div>

                        {/* Text description */}
                        <div className="flex-1 min-w-0">
                          <p
                            className={`font-bold text-label-md transition-colors ${
                              isCompleted
                                ? "text-emerald-700 dark:text-emerald-400"
                                : isProcessing
                                ? "text-primary"
                                : "text-on-surface-variant/60"
                            }`}
                          >
                            {agent.name}
                          </p>
                          <p
                            className={`text-body-md truncate mt-0.5 ${
                              isProcessing ? "text-on-background font-medium" : "text-on-surface-variant/60"
                            }`}
                          >
                            {isProcessing ? (
                              <span className="flex items-center gap-xs">
                                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-ping" />
                                {agent.desc}
                              </span>
                            ) : (
                              agent.desc
                            )}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* RESULTS RENDERING */}
            {result && !loading && (
              <div className="flex flex-col gap-md animate-scale-in">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
                  {/* XGBoost */}
                  <div className="card-hover bg-white dark:bg-zinc-900 border-l-4 border-l-[#ea580c] border border-outline-variant rounded-xl p-lg flex flex-col justify-between h-48 group">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-label-md text-on-surface-variant">Agente ML · R²=91.5%</p>
                        <h4 className="text-headline-md text-primary font-bold">XGBoost</h4>
                      </div>
                      <span className="material-symbols-outlined text-[#ea580c] text-[28px] transition-transform duration-300 group-hover:scale-110"
                        style={{ fontVariationSettings: "'FILL' 1" }}>analytics</span>
                    </div>
                    <div className="flex items-end justify-between">
                      <span className="text-[42px] font-black text-[#ea580c]" style={{ fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
                        {result.prediccion_ml.toFixed(1)}
                      </span>
                      <span className={`${getRisk(result.riesgo_ml).bg} ${getRisk(result.riesgo_ml).text} px-sm py-xs rounded-lg text-label-md border ${getRisk(result.riesgo_ml).border} ${getRisk(result.riesgo_ml).glow}`}>
                        {getRisk(result.riesgo_ml).label}
                      </span>
                    </div>
                  </div>

                  {/* LSTM PyTorch */}
                  {result.prediccion_lstm != null && (
                    <div className="card-hover bg-white dark:bg-zinc-900 border-l-4 border-l-[#0891b2] border border-outline-variant rounded-xl p-lg flex flex-col justify-between h-48 group">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-label-md text-on-surface-variant">Agente Seq. · R²=90.4%</p>
                          <h4 className="text-headline-md text-primary font-bold">LSTM PyTorch</h4>
                        </div>
                        <span className="material-symbols-outlined text-[#0891b2] text-[28px] transition-transform duration-300 group-hover:scale-110"
                          style={{ fontVariationSettings: "'FILL' 1" }}>timeline</span>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-[42px] font-black text-[#0891b2]" style={{ fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
                          {result.prediccion_lstm.toFixed(1)}
                        </span>
                        <span className={`${getRisk(result.riesgo_lstm).bg} ${getRisk(result.riesgo_lstm).text} px-sm py-xs rounded-lg text-label-md border ${getRisk(result.riesgo_lstm).border} ${getRisk(result.riesgo_lstm).glow}`}>
                          {getRisk(result.riesgo_lstm).label}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Consensus Hero Card */}
                <div className="relative overflow-hidden bg-primary text-on-primary rounded-xl p-xl shadow-xl flex flex-col items-center justify-center text-center animate-fade-in-up delay-150">
                  <div className="relative z-10 space-y-md w-full">
                    <div className="flex flex-col items-center gap-xs">
                      <div className="w-14 h-14 bg-secondary-container text-on-secondary-container rounded-full flex items-center justify-center mb-sm shadow-lg">
                        <span className="material-symbols-outlined text-[32px]" style={{ fontVariationSettings: "'FILL' 1" }}>hub</span>
                      </div>
                      <p className="text-label-md text-primary-fixed-dim uppercase tracking-widest font-bold">Consenso Global del Sistema</p>
                      <h4 className="text-headline-lg text-white font-bold">Fusión Ensemble (Agente 5)</h4>
                    </div>

                    <div className="py-sm">
                      <div className="flex items-baseline justify-center gap-sm">
                        <span className="text-[72px] font-black tracking-tighter leading-none" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {result.prediccion_ensemble.toFixed(1)}
                        </span>
                        <span className="text-headline-sm text-surface-variant">casos / 100k hab.</span>
                      </div>
                    </div>

                    <div className="flex flex-col items-center gap-md">
                      <div className={`${getRisk(result.riesgo_ensemble).ensemble} text-white px-lg py-sm rounded-full text-headline-sm font-bold shadow-lg flex items-center gap-sm border border-white/20 ${getRisk(result.riesgo_ensemble).glow}`}>
                        <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>{getRisk(result.riesgo_ensemble).icon}</span>
                        Nivel de Alerta: {getRisk(result.riesgo_ensemble).label}
                      </div>

                      {/* Régimen epidémico (Agente 6) */}
                      {result.regimen_epidemico && (
                        <p className="text-[12px] text-surface-variant italic">
                          {result.regimen_descripcion}
                        </p>
                      )}

                      {/* Barra de pesos dinámica */}
                      {result.ensemble_w_xgb != null && (
                        <div className="w-full max-w-xs space-y-xs">
                          <div className="flex justify-between text-[11px] text-surface-variant font-mono">
                            <span>XGB {Math.round(result.ensemble_w_xgb * 100)}%</span>
                            <span>LSTM {Math.round(result.ensemble_w_lstm * 100)}%</span>
                          </div>
                          <div className="flex h-2 rounded-full overflow-hidden bg-white/10">
                            <div
                              className="bg-blue-400 transition-all duration-500"
                              style={{ width: `${result.ensemble_w_xgb * 100}%` }}
                            />
                            <div
                              className="bg-violet-400 transition-all duration-500"
                              style={{ width: `${result.ensemble_w_lstm * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Technical Note */}
                <div className="bg-surface-container rounded-lg p-md border border-outline-variant/30 flex items-start gap-md text-[12px] leading-relaxed text-on-surface-variant">
                  <span className="material-symbols-outlined text-primary mt-0.5">info</span>
                  <div className="space-y-xs">
                    <p className="text-label-md font-bold text-primary">Nota de Verificación</p>
                    <p>
                      El ensemble combina <strong>XGBoost</strong> (R²=91.5%) y <strong>LSTM PyTorch</strong> (R²=90.4%) con pesos base 50/50, ajustados dinámicamente por el Agente 6 según el régimen epidémico del departamento. La clasificación de riesgo usa percentiles locales calibrados por departamento: Endémico (&le;p50), Alerta (p50–p90), Epidemia (&gt;p90).
                    </p>
                  </div>
                </div>
              </div>
            )}
          </section>
          </div>{/* end inner grid */}
        </div>
      )}

      {activeSubtab === "Histórico" && (
        <div className="space-y-lg animate-fade-in">
          {(!selectedCountry || !selectedDept) ? (
            <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-xl text-center">
              <span className="material-symbols-outlined text-outline text-[48px] mb-sm">location_off</span>
              <h4 className="text-headline-md font-bold text-primary">Sin Departamento Seleccionado</h4>
              <p className="text-on-surface-variant text-body-md mt-xs">
                Seleccione un país y departamento en la pestaña de Simulación para visualizar su serie temporal histórica.
              </p>
            </div>
          ) : (
            <>
              {loadingHistory ? (
                <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-xl text-center space-y-md">
                  <span className="material-symbols-outlined animate-spin text-[36px] text-primary">progress_activity</span>
                  <p className="text-on-surface-variant">Cargando serie histórica departamental...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-lg">
                  {/* SVG Chart — ocupa 7 cols en pantallas grandes */}
                  <div className="lg:col-span-7">
                    <HistoricalChart data={historicalData} />
                  </div>

                  {/* Historical Table — ocupa 5 cols, con scroll vertical */}
                  <div className="lg:col-span-5 bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-lg shadow-sm flex flex-col overflow-hidden">
                    <div className="flex justify-between items-center mb-md flex-shrink-0">
                      <div>
                        <h4 className="text-headline-md font-bold text-primary">Registros Históricos</h4>
                        <p className="text-[11px] text-on-surface-variant mt-0.5">{selectedDept} · {selectedCountry}</p>
                      </div>
                      <span className="text-[11px] font-bold text-primary bg-surface-container px-sm py-xs rounded-lg">
                        {historicalData.length} meses
                      </span>
                    </div>
                    <div className="overflow-y-auto flex-1" style={{ maxHeight: "480px" }}>
                      <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 z-10">
                          <tr className="border-b border-outline-variant/60 text-on-surface-variant font-bold text-[11px] bg-surface-container-low">
                            <th className="py-xs px-sm whitespace-nowrap">Fecha</th>
                            <th className="py-xs px-sm whitespace-nowrap">Casos</th>
                            <th className="py-xs px-sm whitespace-nowrap">Inc./100k</th>
                            <th className="py-xs px-sm whitespace-nowrap">Tmax</th>
                            <th className="py-xs px-sm whitespace-nowrap">Lluvia</th>
                          </tr>
                        </thead>
                        <tbody>
                          {historicalData.slice().reverse().map((rec, idx) => (
                            <tr key={idx} className="border-b border-outline-variant/20 hover:bg-surface-container-lowest text-[12px] transition-colors">
                              <td className="py-xs px-sm font-bold text-on-surface whitespace-nowrap">{rec.fecha}</td>
                              <td className="py-xs px-sm font-semibold text-primary">{rec.casos}</td>
                              <td className="py-xs px-sm text-on-surface-variant">{rec.incidencia}</td>
                              <td className="py-xs px-sm">{rec.tmax}°C</td>
                              <td className="py-xs px-sm">{rec.precipitacion}mm</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {activeSubtab === "Alertas" && (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-lg animate-fade-in text-on-surface">
          {/* Risk Level Percentiles Table */}
          <div className="md:col-span-5 bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-lg shadow-sm">
            <h4 className="text-headline-md font-bold text-primary mb-md">Límites de Alerta Local</h4>
            <p className="text-on-surface-variant text-[12px] leading-relaxed mb-lg">
              Los niveles de riesgo están <strong>calibrados localmente</strong> para el departamento de {selectedDept || "seleccionado"} en base a sus percentiles históricos de incidencia (2014-2022).
            </p>
            <div className="space-y-sm">
              <div className="flex justify-between items-center p-sm bg-emerald-50 dark:bg-emerald-950/20 border-l-4 border-emerald-500 rounded">
                <div>
                  <h5 className="text-[13px] font-bold text-emerald-800 dark:text-emerald-300">Endémico (&le;p50)</h5>
                  <p className="text-[11px] text-emerald-700/80 dark:text-emerald-400">Incidencia dentro del rango endémico habitual.</p>
                </div>
                <span className="font-mono text-label-md font-bold text-emerald-800 dark:text-emerald-300">&le; {localP50.toFixed(2)}</span>
              </div>

              <div className="flex justify-between items-center p-sm bg-orange-50 dark:bg-orange-950/20 border-l-4 border-orange-500 rounded">
                <div>
                  <h5 className="text-[13px] font-bold text-orange-800 dark:text-orange-300">Alerta (p50 – p90)</h5>
                  <p className="text-[11px] text-orange-700/80 dark:text-orange-400">Brotes localizados. Fumigaciones selectivas.</p>
                </div>
                <span className="font-mono text-label-md font-bold text-orange-800 dark:text-orange-300">{localP50.toFixed(2)} — {localP90.toFixed(2)}</span>
              </div>

              <div className="flex justify-between items-center p-sm bg-red-50 dark:bg-red-950/20 border-l-4 border-red-600 rounded">
                <div>
                  <h5 className="text-[13px] font-bold text-red-800 dark:text-red-300">Epidemia (&gt;p90)</h5>
                  <p className="text-[11px] text-red-700/80 dark:text-red-400">Emergencia de salud pública. Acción inmediata.</p>
                </div>
                <span className="font-mono text-label-md font-bold text-red-800 dark:text-red-300">&gt; {localP90.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Epidemiological Action Guide */}
          <div className="md:col-span-7 bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl p-lg shadow-sm space-y-md">
            <h4 className="text-headline-md font-bold text-primary">Protocolo Científico de Alerta Temprana</h4>
            
            <div className="space-y-md text-body-md text-on-surface-variant leading-relaxed">
              <p>
                Basado en el enfoque del modelo de ensamble de la tesis, las fluctuaciones de temperatura y lluvia preceden a los brotes de dengue en aproximadamente <strong>1 a 2 meses (lags climáticos)</strong> debido a la tasa de reproducción vectorial del mosquito <em>Aedes aegypti</em>.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-md pt-xs">
                <div className="p-md bg-surface-container rounded-xl border border-outline-variant/50">
                  <div className="flex items-center gap-xs text-primary font-bold text-label-md mb-xs">
                    <span className="material-symbols-outlined text-[18px]">cleaning_services</span>
                    Lava, Tapa, Voltea, Tira
                  </div>
                  <p className="text-[12px] leading-relaxed">
                    Acción preventiva domiciliaria número uno. Tapar tanques de almacenamiento e inactivar cualquier envase con agua estancada.
                  </p>
                </div>

                <div className="p-md bg-surface-container rounded-xl border border-outline-variant/50">
                  <div className="flex items-center gap-xs text-primary font-bold text-label-md mb-xs">
                    <span className="material-symbols-outlined text-[18px]">medical_services</span>
                    Cerco Epidemiológico
                  </div>
                  <p className="text-[12px] leading-relaxed">
                    Si el ensemble proyecta una incidencia en zona de Alerta/Epidemia, se deben desplegar brigadas de salud en un radio de 500 metros en torno a casos índice.
                  </p>
                </div>

                <div className="p-md bg-surface-container rounded-xl border border-outline-variant/50">
                  <div className="flex items-center gap-xs text-primary font-bold text-label-md mb-xs">
                    <span className="material-symbols-outlined text-[18px]">biotech</span>
                    Monitoreo Larvario
                  </div>
                  <p className="text-[12px] leading-relaxed">
                    Aplicación de larvicidas biológicos (Bti) en canales de drenaje público antes de que comience el periodo húmedo estacional.
                  </p>
                </div>

                <div className="p-md bg-surface-container rounded-xl border border-outline-variant/50">
                  <div className="flex items-center gap-xs text-primary font-bold text-label-md mb-xs">
                    <span className="material-symbols-outlined text-[18px]">campaign</span>
                    Información Local
                  </div>
                  <p className="text-[12px] leading-relaxed">
                    Difusión radial y comunitaria sobre síntomas de dengue grave (fiebre alta, dolor retroocular, sangrados de encías) para evitar automedicación.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Internal component for historical SVG line/bar rendering
function HistoricalChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="text-center py-xl text-on-surface-variant">Sin datos históricos suficientes para graficar.</div>;
  }

  // Slice to last 24 records to display neatly
  const chartData = data.slice(-24);
  const maxInc = Math.max(...chartData.map(d => d.incidencia), 5);
  const maxPrec = Math.max(...chartData.map(d => d.precipitacion), 50);

  const width = 800;
  const height = 280;
  const paddingLeft = 50;
  const paddingRight = 50;
  const paddingTop = 30;
  const paddingBottom = 40;

  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  // Calculate rendering coords
  const points = chartData.map((d, i) => {
    const x = paddingLeft + (i * (chartWidth / (chartData.length - 1)));
    const y = height - paddingBottom - (d.incidencia * (chartHeight / maxInc));
    return { x, y, ...d };
  });

  const polylinePoints = points.map(p => `${p.x},${p.y}`).join(" ");

  return (
    <div className="w-full bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-sm">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-sm mb-lg">
        <div>
          <h4 className="text-headline-md font-bold text-primary">Tendencia de Alerta Vectorial y Lluvias</h4>
          <p className="text-label-md text-on-surface-variant">Serie temporal de los últimos 24 meses</p>
        </div>
        <div className="flex gap-md text-label-md font-semibold">
          <span className="flex items-center gap-xs">
            <span className="w-3 h-3 bg-primary rounded-full"></span>
            Incidencia (línea azul)
          </span>
          <span className="flex items-center gap-xs">
            <span className="w-3 h-3 bg-sky-200 dark:bg-sky-900/60 rounded"></span>
            Lluvia (barras celestes)
          </span>
        </div>
      </div>
      
      <div className="w-full overflow-x-auto">
        <div className="min-w-[650px]">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible select-none">
            {/* Gridlines */}
            {[0, 0.25, 0.5, 0.75, 1].map((ratio, idx) => {
              const y = height - paddingBottom - ratio * chartHeight;
              const valInc = (ratio * maxInc).toFixed(0);
              const valPrec = (ratio * maxPrec).toFixed(0);
              return (
                <g key={idx} className="opacity-30">
                  <line 
                    x1={paddingLeft} 
                    y1={y} 
                    x2={width - paddingRight} 
                    y2={y} 
                    className="stroke-outline"
                    strokeWidth="0.5" 
                    strokeDasharray="4,4" 
                  />
                  {/* Left Axis: Incidencia */}
                  <text x={paddingLeft - 10} y={y + 4} textAnchor="end" className="text-[9px] fill-on-surface font-mono font-bold">{valInc}</text>
                  {/* Right Axis: Precipitación */}
                  <text x={width - paddingRight + 10} y={y + 4} textAnchor="start" className="text-[9px] fill-sky-500 font-mono font-bold">{valPrec}mm</text>
                </g>
              );
            })}

            {/* Precipitation Bars */}
            {points.map((p, idx) => {
              const barWidth = (chartWidth / chartData.length) * 0.45;
              const barHeight = p.precipitacion * (chartHeight / maxPrec);
              const x = p.x - barWidth / 2;
              const y = height - paddingBottom - barHeight;
              return (
                <g key={`bar-${idx}`}>
                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    className="fill-sky-100 dark:fill-sky-950/40 stroke-sky-200 dark:stroke-sky-900/20"
                    strokeWidth="0.5"
                  />
                </g>
              );
            })}

            {/* Incidencia Line */}
            <polyline
              fill="none"
              stroke="#0284c7"
              strokeWidth="3"
              points={polylinePoints}
              className="stroke-primary"
            />

            {/* Markers */}
            {points.map((p, idx) => (
              <g key={`marker-${idx}`} className="group cursor-pointer">
                <circle
                  cx={p.x}
                  cy={p.y}
                  r="4.5"
                  className="fill-primary stroke-white dark:stroke-zinc-900 stroke-2 hover:r-6 transition-all"
                />
                <circle
                  cx={p.x}
                  cy={p.y}
                  r="12"
                  className="fill-transparent cursor-pointer"
                />
                <title>{`Mes: ${p.fecha}\nIncidencia: ${p.incidencia} / 100k\nCasos: ${p.casos} hab\nLluvias: ${p.precipitacion} mm`}</title>
              </g>
            ))}

            {/* X Axis Labels */}
            {points.map((p, idx) => {
              if (idx % 3 !== 0) return null;
              return (
                <text
                  key={`lbl-${idx}`}
                  x={p.x}
                  y={height - paddingBottom + 18}
                  textAnchor="middle"
                  className="text-[9px] fill-on-surface-variant font-medium"
                >
                  {p.fecha}
                </text>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
