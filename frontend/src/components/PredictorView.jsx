import React, { useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Risk badge styling
const riskStyles = {
  Normal: { bg: "bg-[#10b981]/10", text: "text-[#00714d]", border: "border-[#10b981]/20", label: "Bajo/Normal", icon: "check_circle", ensemble: "bg-[#10b981]" },
  Vigilancia: { bg: "bg-[#10b981]/10", text: "text-[#00714d]", border: "border-[#10b981]/20", label: "Vigilancia", icon: "visibility", ensemble: "bg-[#10b981]" },
  Alerta: { bg: "bg-[#ea580c]/10", text: "text-[#ea580c]", border: "border-[#ea580c]/20", label: "Alerta", icon: "warning", ensemble: "bg-[#ea580c]" },
  Epidemia: { bg: "bg-[#ba1a1a]/10", text: "text-[#ba1a1a]", border: "border-[#ba1a1a]/20", label: "Epidemia", icon: "emergency", ensemble: "bg-[#ba1a1a]" },
};

export default function PredictorView({ metadata, selectedCountry, selectedDept, setSelectedCountry, setSelectedDept }) {
  // Location & time
  const [ano, setAno] = useState(2022);
  const [mes, setMes] = useState(6);

  // Climate sliders
  const [tmax, setTmax] = useState(32.5);
  const [tmin, setTmin] = useState(22.0);
  const [humedad, setHumedad] = useState(78);
  const [precip, setPrecip] = useState(120.4);

  // Lag sliders (matching Stitch reference)
  const [lag1, setLag1] = useState(45);

  // Sanitation slider (matching Stitch reference)
  const [aguaPotable, setAguaPotable] = useState(82.3);

  // State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const countries = metadata ? Object.keys(metadata) : [];
  const departments = metadata && selectedCountry ? metadata[selectedCountry]?.departamentos || [] : [];

  useEffect(() => {
    if (departments.length > 0 && !departments.includes(selectedDept)) {
      setSelectedDept(departments[0]);
    }
  }, [selectedCountry, departments]);

  const handleReset = () => {
    setTmax(32.5);
    setTmin(22.0);
    setHumedad(78);
    setPrecip(120.4);
    setLag1(45);
    setAguaPotable(82.3);
    setResult(null);
    setError(null);
  };

  const handlePredict = useCallback(async () => {
    if (!selectedCountry || !selectedDept) {
      setError("Selecciona un país y departamento antes de ejecutar la simulación.");
      return;
    }
    setLoading(true);
    setError(null);

    const body = {
      iso_a0: selectedCountry,
      adm_1_name: selectedDept,
      ano,
      mes,
      clima_overrides: {
        tmax_promedio: tmax,
        tmin_promedio: tmin,
        precipitacion: precip,
        humedad_promedio: humedad,
      },
    };

    try {
      const res = await fetch(`${API_URL}/api/predict/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Error en simulación");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedCountry, selectedDept, ano, mes, tmax, tmin, precip, humedad]);

  const getRisk = (r) => riskStyles[r?.nivel] || riskStyles.Normal;

  // Ensemble variance = |ml - dl| / 2
  const ensembleVariance = result
    ? (Math.abs(result.prediccion_ml - result.prediccion_dl) / 2).toFixed(1)
    : "—";

  // Confidence metric (inversely proportional to model disagreement)
  const confidence = result
    ? Math.max(70, 100 - (Math.abs(result.prediccion_ml - result.prediccion_dl) / Math.max(result.prediccion_ensemble, 1)) * 50).toFixed(1)
    : "—";

  return (
    <div className="w-full max-w-[1440px] mx-auto space-y-lg">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-sm mb-xs">
          <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>sensors</span>
          <span className="text-secondary font-label-md font-bold uppercase tracking-wider">Inferencia en Tiempo Real</span>
        </div>
        <h2 className="text-headline-lg text-primary font-bold">Panel de Simulación e Inferencia en Vivo</h2>
        <p className="text-on-surface-variant text-body-md mt-xs max-w-2xl">
          Ajuste los parámetros climáticos y socioeconómicos para observar cómo el sistema híbrido de IA proyecta la incidencia de dengue.
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-lg">

        {/* ═══════════ LEFT COLUMN: PARAMETERS ═══════════ */}
        <section className="lg:col-span-5 flex flex-col gap-lg">
          <div className="bg-white border border-outline-variant rounded-xl p-lg flex flex-col gap-lg shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">

            {/* Panel Header */}
            <div className="flex items-center justify-between">
              <h3 className="text-headline-md text-primary font-bold">Configuración de Parámetros</h3>
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
                <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">LOCALIZACIÓN</h4>
              </div>
              <div className="grid grid-cols-2 gap-md pt-xs">
                <div className="space-y-xs">
                  <label className="text-body-md text-on-surface-variant">País</label>
                  <select
                    value={selectedCountry}
                    onChange={(e) => setSelectedCountry(e.target.value)}
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
                <div className="space-y-xs">
                  <label className="text-body-md text-on-surface-variant">Año</label>
                  <select
                    value={ano}
                    onChange={(e) => setAno(parseInt(e.target.value))}
                    className="w-full bg-surface-container-high text-primary font-bold text-label-md px-sm py-2 rounded-lg border-none outline-none cursor-pointer"
                  >
                    {[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022].map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-xs">
                  <label className="text-body-md text-on-surface-variant">Mes</label>
                  <select
                    value={mes}
                    onChange={(e) => setMes(parseInt(e.target.value))}
                    className="w-full bg-surface-container-high text-primary font-bold text-label-md px-sm py-2 rounded-lg border-none outline-none cursor-pointer"
                  >
                    {["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"].map((m, i) => (
                      <option key={i + 1} value={i + 1}>{m}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* ─── VARIABLES CLIMÁTICAS ─── */}
            <div className="space-y-md">
              <div className="flex items-center gap-sm border-b border-outline-variant pb-xs">
                <span className="material-symbols-outlined text-primary-fixed-dim">thermostat</span>
                <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">VARIABLES CLIMÁTICAS</h4>
              </div>

              <div className="space-y-lg pt-xs">
                {/* Temp Max */}
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Temperatura Máxima (°C)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {tmax.toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range" min="15" max="45" step="0.1" value={tmax}
                    onChange={(e) => setTmax(parseFloat(e.target.value))}
                    className="slider-custom"
                  />
                </div>

                {/* Temp Min */}
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Temperatura Mínima (°C)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {tmin.toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range" min="10" max="30" step="0.1" value={tmin}
                    onChange={(e) => setTmin(parseFloat(e.target.value))}
                    className="slider-custom"
                  />
                </div>

                {/* Humedad */}
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Humedad Relativa (%)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {humedad}%
                    </span>
                  </div>
                  <input
                    type="range" min="0" max="100" step="1" value={humedad}
                    onChange={(e) => setHumedad(parseInt(e.target.value))}
                    className="slider-custom"
                  />
                </div>

                {/* Precipitación */}
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Precipitación (mm)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {precip.toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range" min="0" max="500" step="0.5" value={precip}
                    onChange={(e) => setPrecip(parseFloat(e.target.value))}
                    className="slider-custom"
                  />
                </div>
              </div>
            </div>

            {/* ─── REZAGOS TEMPORALES ─── */}
            <div className="space-y-md">
              <div className="flex items-center gap-sm border-b border-outline-variant pb-xs">
                <span className="material-symbols-outlined text-primary-fixed-dim">history</span>
                <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">REZAGOS TEMPORALES (LAGS)</h4>
              </div>
              <div className="space-y-lg pt-xs">
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Casos mes anterior (t-1)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {lag1}
                    </span>
                  </div>
                  <input
                    type="range" min="0" max="200" step="1" value={lag1}
                    onChange={(e) => setLag1(parseInt(e.target.value))}
                    className="slider-custom"
                  />
                </div>
              </div>
            </div>

            {/* ─── ACCESO A SANEAMIENTO ─── */}
            <div className="space-y-md">
              <div className="flex items-center gap-sm border-b border-outline-variant pb-xs">
                <span className="material-symbols-outlined text-primary-fixed-dim">water_drop</span>
                <h4 className="text-label-md font-bold text-on-surface uppercase tracking-wider">ACCESO A SANEAMIENTO</h4>
              </div>
              <div className="space-y-lg pt-xs">
                <div className="space-y-xs">
                  <div className="flex justify-between items-center">
                    <label className="text-body-md text-on-surface-variant">Agua potable básica (%)</label>
                    <span className="font-bold text-primary bg-surface-container-high px-sm py-0.5 rounded-lg text-label-md" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {aguaPotable.toFixed(1)}%
                    </span>
                  </div>
                  <input
                    type="range" min="0" max="100" step="0.1" value={aguaPotable}
                    onChange={(e) => setAguaPotable(parseFloat(e.target.value))}
                    className="slider-custom"
                  />
                </div>
              </div>
            </div>

            {/* ─── EJECUTAR ─── */}
            <button
              onClick={handlePredict}
              disabled={loading || !selectedCountry || !selectedDept}
              className="w-full bg-primary text-on-primary px-md py-3 rounded-xl text-label-md font-bold hover:opacity-90 transition-all flex items-center justify-center gap-sm cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <span className="material-symbols-outlined animate-spin text-[18px]">progress_activity</span>
                  Procesando inferencia...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-[18px]">rocket_launch</span>
                  Ejecutar Predicción Multi-Agente
                </>
              )}
            </button>
          </div>
        </section>

        {/* ═══════════ RIGHT COLUMN: RESULTS ═══════════ */}
        <section className="lg:col-span-7 flex flex-col gap-lg">

          {/* Results Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-headline-md text-primary font-bold">Resultados de Inferencia <span className="text-on-surface-variant font-normal text-body-md">(Casos por 100k hab.)</span></h3>
            <div className="flex items-center gap-xs">
              <span className="w-3 h-3 bg-secondary rounded-full animate-pulse"></span>
              <span className="text-label-md text-secondary font-medium">Sistema en Vivo</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-error-container border border-outline-variant rounded-lg p-md flex items-center gap-md">
              <span className="material-symbols-outlined text-on-error-container">error</span>
              <p className="text-label-md text-on-error-container font-medium">{error}</p>
            </div>
          )}

          {/* ─── PLACEHOLDER (no results yet) ─── */}
          {!result && !loading && !error && (
            <div className="flex flex-col gap-md">
              {/* Mock cards - grayed out */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
                <div className="bg-white border-l-4 border-l-outline-variant border border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 opacity-50">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-label-md text-on-surface-variant">Predicción Agente ML</p>
                      <h4 className="text-headline-md text-primary font-bold">XGBoost</h4>
                    </div>
                    <span className="material-symbols-outlined text-outline">analytics</span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-[48px] font-black text-outline" style={{ fontVariantNumeric: "tabular-nums" }}>—</span>
                    <span className="bg-surface-container text-on-surface-variant px-sm py-xs rounded-lg text-label-md">Esperando...</span>
                  </div>
                </div>
                <div className="bg-white border-l-4 border-l-outline-variant border border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 opacity-50">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-label-md text-on-surface-variant">Predicción Agente DL</p>
                      <h4 className="text-headline-md text-primary font-bold">MLP PyTorch</h4>
                    </div>
                    <span className="material-symbols-outlined text-outline">neurology</span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-[48px] font-black text-outline" style={{ fontVariantNumeric: "tabular-nums" }}>—</span>
                    <span className="bg-surface-container text-on-surface-variant px-sm py-xs rounded-lg text-label-md">Esperando...</span>
                  </div>
                </div>
              </div>
              {/* Mock ensemble */}
              <div className="bg-primary/10 border border-outline-variant rounded-xl p-xl flex flex-col items-center justify-center text-center opacity-40" style={{ minHeight: "280px" }}>
                <div className="w-12 h-12 bg-surface-container rounded-full flex items-center justify-center mb-md">
                  <span className="material-symbols-outlined text-[32px] text-outline">hub</span>
                </div>
                <p className="text-label-md text-on-surface-variant uppercase tracking-widest font-bold">Consenso Global del Sistema</p>
                <p className="text-headline-md text-on-surface-variant mt-sm">Configura los parámetros y ejecuta la predicción</p>
              </div>
            </div>
          )}

          {/* ─── LOADING SKELETON ─── */}
          {loading && (
            <div className="flex flex-col gap-md">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
                {[1, 2].map((i) => (
                  <div key={i} className="bg-white border border-outline-variant rounded-lg p-lg h-48">
                    <div className="h-4 w-32 shimmer rounded mb-sm"></div>
                    <div className="h-6 w-24 shimmer rounded mb-lg"></div>
                    <div className="h-12 w-20 shimmer rounded mt-auto"></div>
                  </div>
                ))}
              </div>
              <div className="bg-primary rounded-xl p-xl flex flex-col items-center" style={{ minHeight: "280px" }}>
                <div className="w-12 h-12 bg-white/10 rounded-full shimmer mb-md"></div>
                <div className="h-6 w-48 bg-white/10 rounded shimmer mb-md"></div>
                <div className="h-20 w-32 bg-white/10 rounded shimmer"></div>
              </div>
            </div>
          )}

          {/* ─── LIVE RESULTS ─── */}
          {result && (
            <div className="flex flex-col gap-md animate-fade-in-up">
              {/* Two Model Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-md">

                {/* XGBoost Card */}
                <div className="bg-white border-l-4 border-l-[#ea580c] border-y border-r border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-label-md text-on-surface-variant">Predicción Agente ML</p>
                      <h4 className="text-headline-md text-primary font-bold">XGBoost</h4>
                    </div>
                    <span className="material-symbols-outlined text-[#ea580c]">analytics</span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-[48px] font-black text-[#ea580c]" style={{ fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
                      {result.prediccion_ml.toFixed(1)}
                    </span>
                    <span className={`${getRisk(result.riesgo_ml).bg} ${getRisk(result.riesgo_ml).text} px-sm py-xs rounded-lg text-label-md border ${getRisk(result.riesgo_ml).border}`}>
                      Riesgo: {getRisk(result.riesgo_ml).label}
                    </span>
                  </div>
                </div>

                {/* MLP PyTorch Card */}
                <div className="bg-white border-l-4 border-l-[#8b5cf6] border-y border-r border-outline-variant rounded-lg p-lg flex flex-col justify-between h-48 hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-label-md text-on-surface-variant">Predicción Agente DL</p>
                      <h4 className="text-headline-md text-primary font-bold">MLP PyTorch</h4>
                    </div>
                    <span className="material-symbols-outlined text-[#8b5cf6]">neurology</span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-[48px] font-black text-[#8b5cf6]" style={{ fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
                      {result.prediccion_dl.toFixed(1)}
                    </span>
                    <span className={`${getRisk(result.riesgo_dl).bg} ${getRisk(result.riesgo_dl).text} px-sm py-xs rounded-lg text-label-md border ${getRisk(result.riesgo_dl).border}`}>
                      Riesgo: {getRisk(result.riesgo_dl).label}
                    </span>
                  </div>
                </div>
              </div>

              {/* ═══ ENSEMBLE HERO CARD ═══ */}
              <div className="relative overflow-hidden bg-primary text-on-primary rounded-xl p-xl shadow-xl flex flex-col items-center justify-center text-center">
                <div className="relative z-10 space-y-md">
                  {/* Icon + Title */}
                  <div className="flex flex-col items-center gap-xs">
                    <div className="w-12 h-12 bg-secondary-container text-on-secondary-container rounded-full flex items-center justify-center mb-sm">
                      <span className="material-symbols-outlined text-[32px]">hub</span>
                    </div>
                    <p className="text-label-md text-primary-fixed-dim uppercase tracking-widest font-bold">Consenso Global del Sistema</p>
                    <h4 className="text-headline-lg text-white font-bold">Fusión Ensemble Promedio (Agente 5)</h4>
                  </div>

                  {/* Giant Number */}
                  <div className="py-md">
                    <div className="flex items-baseline justify-center gap-sm">
                      <span className="text-[84px] font-black tracking-tighter leading-none" style={{ fontVariantNumeric: "tabular-nums" }}>
                        {result.prediccion_ensemble.toFixed(1)}
                      </span>
                      <span className="text-headline-md text-surface-variant">casos</span>
                    </div>
                  </div>

                  {/* Risk Badge + Stats */}
                  <div className="flex flex-col items-center gap-md">
                    <div className={`${getRisk(result.riesgo_ensemble).ensemble} text-white px-lg py-sm rounded-full text-headline-md font-bold shadow-lg flex items-center gap-sm border border-white/20`}>
                      <span className="material-symbols-outlined">{getRisk(result.riesgo_ensemble).icon}</span>
                      Nivel de Alerta: {getRisk(result.riesgo_ensemble).label}
                    </div>

                    <div className="flex gap-lg mt-sm border-t border-white/10 pt-lg w-full max-w-sm">
                      <div className="flex-1">
                        <p className="text-primary-fixed-dim text-label-md">Confianza</p>
                        <p className="text-headline-md font-bold text-white">{confidence}%</p>
                      </div>
                      <div className="w-px bg-white/10"></div>
                      <div className="flex-1">
                        <p className="text-primary-fixed-dim text-label-md">Varianza</p>
                        <p className="text-headline-md font-bold text-white">±{ensembleVariance}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical Note */}
              <div className="bg-surface-container rounded-lg p-md border border-outline-variant/30 flex items-start gap-md">
                <span className="material-symbols-outlined text-primary mt-0.5">info</span>
                <div className="space-y-xs">
                  <p className="text-label-md font-bold text-primary">Nota Técnica</p>
                  <p className="text-body-md text-on-surface-variant">
                    La fusión utiliza una media ponderada basada en el error cuadrático medio (MSE) histórico de los últimos 24 meses. Los pesos actuales son XGBoost (40%) y MLP (60%).
                    La clasificación de riesgo se basa en percentiles calibrados del dataset histórico 2014-2022:
                    Normal (&lt;p25), Vigilancia (p25-p50), Alerta (p50-p90), Epidemia (&gt;p90).
                    {selectedDept && <> Departamento seleccionado: <strong>{selectedDept}</strong>, {selectedCountry}.</>}
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
