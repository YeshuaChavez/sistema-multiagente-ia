import React from "react";

const agents = [
  {
    id: 1,
    name: "Agente de Ingesta de Datos",
    icon: "cloud_download",
    color: "bg-sky-500",
    colorLight: "bg-sky-50",
    description:
      "Se encarga de la recolección y consolidación automática de datos epidemiológicos (OpenDengue) y climáticos (NASA POWER). Consulta fuentes externas periódicamente y las estructura en formato tabular estandarizado.",
    tech: ["OpenDengue API", "NASA POWER API", "Pandas", "Requests"],
    input: "Fuentes externas (APIs REST públicas)",
    output: "CSVs estructurados → datos_crudos/",
  },
  {
    id: 2,
    name: "Agente de Preprocesamiento",
    icon: "filter_alt",
    color: "bg-indigo-500",
    colorLight: "bg-indigo-50",
    description:
      "Limpia, normaliza y transforma los datos brutos. Calcula tasas de incidencia por 100,000 habitantes, genera variables de rezago temporal (lag-1, lag-2, lag-3), codifica la estacionalidad cíclica (sin/cos) y fusiona datos climáticos con epidemiológicos.",
    tech: ["Pandas", "NumPy", "Scikit-Learn (StandardScaler)"],
    input: "CSVs crudos de datos_crudos/",
    output: "Dataset maestro → datos_procesados/dataset_ml_final.csv",
  },
  {
    id: 3,
    name: "Agente de Machine Learning (XGBoost)",
    icon: "precision_manufacturing",
    color: "bg-orange-500",
    colorLight: "bg-orange-50",
    description:
      "Entrena y ejecuta un modelo XGBoost (Gradient Boosting basado en árboles) para la predicción de la tasa de incidencia de dengue. Utiliza hiperparámetros optimizados mediante GridSearchCV + TimeSeriesSplit (n_estimators=800, lr=0.01, max_depth=4) y genera importancias SHAP locales y globales mediante TreeSHAP.",
    tech: ["XGBoost", "Scikit-Learn", "SHAP (TreeSHAP)", "Joblib"],
    input: "Dataset preprocesado + 73 features ingenieriles",
    output: "Predicción ML (R²=91.5%) + SHAP values",
  },
  {
    id: 4,
    name: "Agente de Deep Learning (LSTM PyTorch)",
    icon: "neurology",
    color: "bg-purple-600",
    colorLight: "bg-purple-50",
    description:
      "Implementa una red neuronal LSTM (Long Short-Term Memory) con PyTorch para capturar dependencias temporales de largo plazo en los datos epidemiológicos. Utiliza 12 meses de lookback, capas recurrentes con dropout y entrenamiento con Adam optimizer + early stopping (ReduceLROnPlateau).",
    tech: ["PyTorch (CPU)", "torch.nn (LSTM)", "Adam Optimizer", "Early Stopping"],
    input: "Secuencias temporales de 12 meses (clima + incidencia)",
    output: "Predicción DL (R²=90.4%) con patrón estacional",
  },
  {
    id: 5,
    name: "Agente de Consenso (Ensemble)",
    icon: "hub",
    color: "bg-emerald-600",
    colorLight: "bg-emerald-50",
    description:
      "Combina las predicciones de los Agentes 3 y 4 usando pesos ajustados dinámicamente por el Agente 6. En régimen normal usa pesos iguales (w_XGB=0.50, w_LSTM=0.50). Clasifica la predicción final en niveles de riesgo calibrados con percentiles históricos por departamento.",
    tech: ["NumPy", "Percentiles Calibrados (p25, p50, p90)"],
    input: "Predicciones de Agente 3 (ML) + Agente 4 (DL) + Pesos de Agente 6",
    output: "Predicción final + Nivel de riesgo (Normal/Vigilancia/Alerta/Epidemia)",
  },
  {
    id: 6,
    name: "Agente de Detección de Régimen Epidémico",
    icon: "crisis_alert",
    color: "bg-rose-600",
    colorLight: "bg-rose-50",
    description:
      "Clasifica el estado epidémico actual en uno de cinco regímenes (Normal, Vigilancia, Pre-brote, Brote activo, Post-pico) usando percentiles históricos locales y la tendencia de incidencia. Ajusta dinámicamente los pesos del ensemble: en Brote activo el LSTM domina (hasta 80%) para capturar momentum; en Post-pico el XGBoost domina porque la regresión a la media es correcta.",
    tech: ["NumPy", "Percentiles Locales por Departamento", "Detección de Tendencia"],
    input: "incidencia_lag1 (escala real) + tendencia log + percentiles p25/p50/p90 locales",
    output: "Régimen epidémico + pesos adaptativos w_XGB / w_LSTM para el Agente 5",
  },
];

const techStack = [
  { category: "Backend", items: ["FastAPI", "Python 3.10+", "Uvicorn", "Pydantic"] },
  { category: "ML / DL", items: ["XGBoost", "PyTorch", "Scikit-Learn", "SHAP"] },
  { category: "Datos", items: ["Pandas", "NumPy", "OpenDengue", "NASA POWER"] },
  { category: "Frontend", items: ["React 19", "Vite", "TailwindCSS", "Leaflet.js"] },
];

const getAgentDelay = (id) => {
  const mapping = { 1: 0, 2: 0.9, 3: 1.8, 4: 2.7, 5: 3.6, 6: 4.5 };
  return `${mapping[id] || 0}s`;
};

const desktopGridPlacements = {
  1: "col-start-1 row-start-1",
  2: "col-start-3 row-start-1",
  3: "col-start-5 row-start-1",
  4: "col-start-5 row-start-3",
  5: "col-start-3 row-start-3",
  6: "col-start-1 row-start-3",
};

export default function InfoView() {
  return (
    <div className="max-w-[1400px] mx-auto space-y-lg">
      {/* Header */}
      <div className="flex items-center gap-md mb-md">
        <div className="w-12 h-12 rounded-xl bg-primary-container flex items-center justify-center">
          <span className="material-symbols-outlined text-on-primary text-[24px]">info</span>
        </div>
        <div>
          <h2 className="text-xl font-bold text-primary">Información Técnica del Sistema</h2>
          <p className="text-label-md text-on-surface-variant">
            Arquitectura Multi-Agente SMA-ML/DL para predicción epidemiológica
          </p>
        </div>
      </div>

      {/* Project Overview */}
      <div className="custom-card rounded-xl p-lg border-l-4 border-l-primary-container">
        <div className="flex items-start gap-md">
          <span className="material-symbols-outlined text-primary-container text-[28px] mt-0.5">school</span>
          <div>
            <h3 className="text-lg font-bold text-primary mb-sm">Proyecto de Investigación Académica</h3>
            <p className="text-[13px] text-on-surface leading-relaxed">
              El <span className="font-bold">Sistema Multi-Agente SMA-ML/DL</span> es un artefacto de software
              desarrollado como Proyecto Final de Investigación. Su objetivo es proporcionar una herramienta de
              <span className="font-semibold"> alerta temprana</span> que anticipe la tasa de incidencia de dengue
              (casos por 100,000 habitantes) a escala subnacional en América Latina, utilizando técnicas de
              Machine Learning y Deep Learning sobre datos epidemiológicos y climáticos del periodo 2014-2022.
            </p>
          </div>
        </div>
      </div>

      {/* Agents Grid */}
      <div>
        <h3 className="text-label-md font-bold text-primary uppercase tracking-wider mb-md flex items-center gap-sm">
          <span className="material-symbols-outlined text-[18px]">smart_toy</span>
          Los 6 Agentes del Sistema
        </h3>
        <div className="space-y-md">
          {agents.map((agent, i) => (
            <div
              key={agent.id}
              className="custom-card rounded-xl p-md sm:p-lg animate-fade-in-up group"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              {/* Mobile: cabecera con icono + badge en fila, luego contenido abajo */}
              <div className="flex items-center gap-md mb-md sm:hidden">
                <div className={`w-12 h-12 rounded-xl ${agent.color} flex items-center justify-center text-white flex-shrink-0 shadow-md`}>
                  <span className="material-symbols-outlined text-[24px]" style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
                </div>
                <div>
                  <span className="text-[9px] font-bold text-white px-2 py-0.5 rounded-full bg-primary-container">
                    AGENTE {agent.id}
                  </span>
                  <h4 className="text-[14px] font-bold text-primary mt-[2px] leading-tight">{agent.name}</h4>
                </div>
              </div>

              {/* Desktop: layout original horizontal */}
              <div className="hidden sm:flex items-start gap-lg">
                <div className={`w-14 h-14 rounded-xl ${agent.color} flex items-center justify-center text-white flex-shrink-0 shadow-md transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}>
                  <span className="material-symbols-outlined text-[28px]" style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-sm mb-sm">
                    <span className="text-[10px] font-bold text-white px-2 py-0.5 rounded-full bg-primary-container">
                      AGENTE {agent.id}
                    </span>
                    <h4 className="text-[15px] font-bold text-primary">{agent.name}</h4>
                  </div>
                  <p className="text-[13px] text-on-surface-variant leading-relaxed mb-md">{agent.description}</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-sm mb-md">
                    <div className="flex items-start gap-xs">
                      <span className="material-symbols-outlined text-sky-500 text-[16px] mt-0.5">input</span>
                      <div>
                        <p className="text-[10px] font-bold text-on-surface-variant uppercase">Entrada</p>
                        <p className="text-[12px] text-on-surface">{agent.input}</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-xs">
                      <span className="material-symbols-outlined text-emerald-500 text-[16px] mt-0.5">output</span>
                      <div>
                        <p className="text-[10px] font-bold text-on-surface-variant uppercase">Salida</p>
                        <p className="text-[12px] text-on-surface">{agent.output}</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-xs">
                    {agent.tech.map((t) => (
                      <span key={t} className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${agent.colorLight} text-on-surface-variant transition-all duration-150 hover:scale-105 hover:shadow-sm cursor-default`}>{t}</span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Mobile: contenido completo debajo del header */}
              <div className="sm:hidden">
                <p className="text-[12px] text-on-surface-variant leading-relaxed mb-md">{agent.description}</p>
                <div className="grid grid-cols-1 gap-xs mb-md">
                  <div className="flex items-start gap-xs bg-surface-container rounded-lg p-xs">
                    <span className="material-symbols-outlined text-sky-500 text-[14px] mt-0.5 flex-shrink-0">input</span>
                    <div>
                      <p className="text-[9px] font-bold text-on-surface-variant uppercase">Entrada</p>
                      <p className="text-[11px] text-on-surface">{agent.input}</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-xs bg-surface-container rounded-lg p-xs">
                    <span className="material-symbols-outlined text-emerald-500 text-[14px] mt-0.5 flex-shrink-0">output</span>
                    <div>
                      <p className="text-[9px] font-bold text-on-surface-variant uppercase">Salida</p>
                      <p className="text-[11px] text-on-surface">{agent.output}</p>
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-xs">
                  {agent.tech.map((t) => (
                     <span key={t} className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${agent.colorLight} text-on-surface-variant cursor-default`}>{t}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Fundamentos Matemáticos ── */}
      <div>
        <h3 className="text-label-md font-bold text-primary uppercase tracking-wider mb-md flex items-center gap-sm">
          <span className="material-symbols-outlined text-[18px]">functions</span>
          Fundamentos Matemáticos del Sistema
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-lg">

          {/* Fórmula 1: Tasa de Incidencia */}
          <div className="custom-card rounded-xl p-lg animate-fade-in-up">
            <div className="flex items-center gap-sm mb-md">
              <div className="w-9 h-9 rounded-lg bg-sky-100 dark:bg-sky-900/30 flex items-center justify-center flex-shrink-0">
                <span className="material-symbols-outlined text-sky-600 dark:text-sky-400 text-[20px]">calculate</span>
              </div>
              <h4 className="text-[14px] font-bold text-primary">Tasa de Incidencia Normalizada</h4>
            </div>
            <p className="text-[12px] text-on-surface-variant mb-md leading-relaxed">
              Métrica estándar de salud pública que permite comparar la carga de enfermedad entre departamentos de distinta densidad demográfica.
            </p>
            {/* Fórmula principal */}
            <div className="bg-surface-container rounded-xl px-lg py-md mb-md font-mono text-center border border-outline-variant/50">
              <div className="text-[15px] text-primary font-bold mb-xs">
                I<sub className="text-[11px]">t</sub> ={" "}
                <span className="inline-flex flex-col items-center mx-xs">
                  <span className="border-b border-on-surface px-xs text-on-surface text-[13px]">
                    casos<sub className="text-[10px]">dengue</sub>(t)
                  </span>
                  <span className="text-on-surface text-[13px] pt-[2px]">
                    población(t)
                  </span>
                </span>
                × 100,000
              </div>
            </div>
            {/* Leyenda */}
            <div className="space-y-xs text-[11px]">
              {[
                ["I(t)", "Tasa de incidencia mensual en el período t"],
                ["casos(t)", "Casos de dengue reportados en el mes t"],
                ["población(t)", "Población total del departamento en el año t"],
              ].map(([sym, desc]) => (
                <div key={sym} className="flex items-start gap-sm">
                  <code className="font-mono font-bold text-primary bg-surface-container px-xs py-[1px] rounded text-[11px] flex-shrink-0">{sym}</code>
                  <span className="text-on-surface-variant">{desc}</span>
                </div>
              ))}
            </div>
            <div className="mt-md pt-md border-t border-outline-variant/40 text-[11px] text-on-surface-variant">
              Unidad de medida: <span className="font-bold text-on-surface">casos / 100,000 habitantes-mes</span>
            </div>
          </div>

          {/* Fórmula 2: Ensemble Ponderado */}
          <div className="custom-card rounded-xl p-lg animate-fade-in-up" style={{ animationDelay: "80ms" }}>
            <div className="flex items-center gap-sm mb-md">
              <div className="w-9 h-9 rounded-lg bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center flex-shrink-0">
                <span className="material-symbols-outlined text-emerald-600 dark:text-emerald-400 text-[20px]">hub</span>
              </div>
              <h4 className="text-[14px] font-bold text-primary">Ensemble Ponderado por R²</h4>
            </div>
            <p className="text-[12px] text-on-surface-variant mb-md leading-relaxed">
              Combina las predicciones del Agente 3 (XGBoost) y el Agente 4 (LSTM) con pesos proporcionales al R² individual de cada modelo en el conjunto de prueba.
            </p>
            {/* Fórmula predicción */}
            <div className="bg-surface-container rounded-xl px-lg py-md mb-sm font-mono border border-outline-variant/50">
              <div className="text-[14px] text-primary font-bold text-center">
                ŷ<sub className="text-[10px]">ens</sub> = w<sub className="text-[10px]">XGB</sub> · ŷ<sub className="text-[10px]">XGB</sub>{" "}
                + w<sub className="text-[10px]">LSTM</sub> · ŷ<sub className="text-[10px]">LSTM</sub>
              </div>
            </div>
            {/* Fórmula pesos */}
            <div className="bg-surface-container rounded-xl px-lg py-sm mb-md font-mono border border-outline-variant/50 text-center text-[12px]">
              <span className="text-on-surface">
                w<sub className="text-[10px]">XGB</sub> ={" "}
                <span className="inline-flex flex-col items-center mx-xs align-middle">
                  <span className="border-b border-on-surface px-xs text-[11px]">R²<sub className="text-[9px]">XGB</sub></span>
                  <span className="text-[11px] pt-[2px]">R²<sub className="text-[9px]">XGB</sub> + R²<sub className="text-[9px]">LSTM</sub></span>
                </span>
                {" "}= <strong className="text-primary">0.5117</strong>
                {"  "}·{"  "}
                w<sub className="text-[10px]">LSTM</sub> = <strong className="text-primary">0.4883</strong>
              </span>
            </div>
            {/* Valores actuales */}
            <div className="grid grid-cols-3 gap-sm text-center">
              {[
                { label: "R² XGBoost", val: "91.49%", color: "text-orange-600 dark:text-orange-400" },
                { label: "R² LSTM", val: "90.35%", color: "text-purple-600 dark:text-purple-400" },
                { label: "R² Ensemble", val: "91.47%", color: "text-emerald-600 dark:text-emerald-400" },
              ].map((m) => (
                <div key={m.label} className="bg-surface-container-low rounded-lg p-xs">
                  <p className={`text-[14px] font-bold tabular ${m.color}`}>{m.val}</p>
                  <p className="text-[10px] text-on-surface-variant">{m.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Fórmula 3: Rezagos (Lags) */}
          <div className="custom-card rounded-xl p-lg animate-fade-in-up" style={{ animationDelay: "160ms" }}>
            <div className="flex items-center gap-sm mb-md">
              <div className="w-9 h-9 rounded-lg bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center flex-shrink-0">
                <span className="material-symbols-outlined text-orange-600 dark:text-orange-400 text-[20px]">history</span>
              </div>
              <h4 className="text-[14px] font-bold text-primary">Variables de Rezago Temporal (Lags)</h4>
            </div>
            <p className="text-[12px] text-on-surface-variant mb-md leading-relaxed">
              Codifican la dependencia temporal: el valor de una variable climática o epidemiológica k meses antes del mes de predicción.
            </p>
            <div className="bg-surface-container rounded-xl px-lg py-md mb-md font-mono border border-outline-variant/50 text-center">
              <div className="text-[14px] text-primary font-bold">
                X<sub className="text-[10px]">lag-k</sub>(t) = X(t − k)
              </div>
              <div className="text-[11px] text-on-surface-variant mt-xs">
                k ∈ &#123;1, 2, 3, 4, 5, 6&#125; meses para variables climáticas
              </div>
              <div className="text-[11px] text-on-surface-variant">
                k ∈ &#123;1, ..., 12&#125; meses para incidencia autorregresiva
              </div>
            </div>
            <div className="space-y-xs text-[11px]">
              {[
                ["tmax_lag3", "Temperatura máxima de hace 3 meses → período de incubación extrínseca del virus en el vector"],
                ["incidencia_lag1", "Casos del mes anterior → componente autorregresiva más predictiva del modelo"],
              ].map(([ex, desc]) => (
                <div key={ex} className="flex items-start gap-sm p-xs bg-surface-container-low rounded-lg">
                  <code className="font-mono font-bold text-orange-600 dark:text-orange-400 text-[11px] flex-shrink-0">{ex}</code>
                  <span className="text-on-surface-variant">{desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Fórmula 4: Codificación Cíclica */}
          <div className="custom-card rounded-xl p-lg animate-fade-in-up" style={{ animationDelay: "240ms" }}>
            <div className="flex items-center gap-sm mb-md">
              <div className="w-9 h-9 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center flex-shrink-0">
                <span className="material-symbols-outlined text-indigo-600 dark:text-indigo-400 text-[20px]">calendar_month</span>
              </div>
              <h4 className="text-[14px] font-bold text-primary">Codificación Cíclica Estacional</h4>
            </div>
            <p className="text-[12px] text-on-surface-variant mb-md leading-relaxed">
              Transforma el mes (1–12) en dos coordenadas continuas sobre el círculo unitario, preservando la distancia temporal correcta entre diciembre y enero.
            </p>
            <div className="bg-surface-container rounded-xl px-lg py-md mb-md font-mono border border-outline-variant/50 text-center space-y-xs">
              <div className="text-[14px] text-primary font-bold">
                mes_sin = sin<span className="font-normal">(</span>
                <span className="inline-flex flex-col items-center mx-[2px] align-middle">
                  <span className="border-b border-on-surface px-xs text-[12px]">2π · mes</span>
                  <span className="text-[12px] pt-[2px]">12</span>
                </span>
                <span className="font-normal">)</span>
              </div>
              <div className="text-[14px] text-primary font-bold">
                mes_cos = cos<span className="font-normal">(</span>
                <span className="inline-flex flex-col items-center mx-[2px] align-middle">
                  <span className="border-b border-on-surface px-xs text-[12px]">2π · mes</span>
                  <span className="text-[12px] pt-[2px]">12</span>
                </span>
                <span className="font-normal">)</span>
              </div>
            </div>
            <div className="text-[11px] text-on-surface-variant leading-relaxed">
              Sin esta codificación, el modelo trataría diciembre (mes 12) como muy lejano de enero (mes 1), distorsionando la estacionalidad anual del dengue.
            </div>
          </div>

        </div>
      </div>

      {/* Architecture Flow */}
      <div className="custom-card rounded-xl p-lg overflow-hidden">
        <style>{`
          @keyframes agentGlow {
            0%, 100% { box-shadow: none; transform: scale(1) translateY(0); filter: brightness(1); }
            8%  { box-shadow: 0 0 0 3px rgba(var(--color-primary), 0.25), 0 8px 24px rgba(var(--color-primary), 0.15); transform: scale(1.04) translateY(-2px); filter: brightness(1.05); }
            16% { box-shadow: none; transform: scale(1) translateY(0); filter: brightness(1); }
          }
          @keyframes dotHRight {
            0%, 100% { left: 0px; opacity: 0; }
            2%  { left: 0px; opacity: 1; }
            14% { left: calc(100% - 10px); opacity: 1; }
            16% { left: calc(100% - 10px); opacity: 0; }
          }
          @keyframes dotHLeft {
            0%, 100% { right: 0px; opacity: 0; }
            2%  { right: 0px; opacity: 1; }
            14% { right: calc(100% - 10px); opacity: 1; }
            16% { right: calc(100% - 10px); opacity: 0; }
          }
          @keyframes dotVDown {
            0%, 100% { top: 0px; opacity: 0; }
            2%  { top: 0px; opacity: 1; }
            14% { top: calc(100% - 10px); opacity: 1; }
            16% { top: calc(100% - 10px); opacity: 0; }
          }
          @keyframes lineFlash {
            0%, 100% { opacity: 0.25; }
            8% { opacity: 0.8; }
          }
          .flow-dot {
            width: 10px; height: 10px;
            background: rgb(var(--color-primary));
            box-shadow: 0 0 0 2px rgba(var(--color-primary), 0.25), 0 0 12px 4px rgba(var(--color-primary), 0.45);
          }
          .dark .flow-dot {
            background: white;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.2), 0 0 12px 4px rgba(255,255,255,0.6);
          }
        `}</style>

        <h3 className="text-label-md font-bold text-primary uppercase tracking-wider mb-lg flex items-center gap-sm">
          <span className="material-symbols-outlined text-[18px]">account_tree</span>
          Flujo de la Arquitectura
        </h3>

        {/* Mobile: columna vertical animada */}
        <div className="flex flex-col items-stretch sm:hidden">
          {agents.map((agent, idx) => (
            <React.Fragment key={agent.id}>
              <div
                className={`flex items-center gap-sm px-4 py-3 rounded-xl ${agent.color} text-white shadow-md`}
                style={{ animation: "agentGlow 5s ease-in-out infinite", animationDelay: getAgentDelay(agent.id) }}
              >
                <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-wider opacity-80">Agente {agent.id}</p>
                  <p className="text-[12px] font-bold leading-tight">{agent.name.split("(")[0].trim().split("Agente de ").pop()}</p>
                </div>
              </div>
              {idx < agents.length - 1 && (
                <div className="flex justify-center">
                  <div className="relative flex flex-col items-center" style={{ height: "30px", width: "12px" }}>
                    <div
                      className="absolute left-1/2 -translate-x-1/2 w-px h-full bg-outline/30"
                      style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: `${(idx * 0.9 + 0.45).toFixed(2)}s` }}
                    />
                    <div
                      className="absolute flow-dot rounded-full"
                      style={{
                        left: "50%", transform: "translateX(-50%)",
                        animation: "dotVDown 5s ease-in-out infinite",
                        animationDelay: `${(idx * 0.9 + 0.45).toFixed(2)}s`,
                      }}
                    />
                  </div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Desktop: 2 filas × 3 columnas animadas en Grid CSS */}
        <div className="hidden sm:grid grid-cols-[1fr_auto_1fr_auto_1fr] gap-y-sm items-center justify-items-center w-full max-w-4xl mx-auto relative">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`group/box flex items-center gap-sm px-5 py-3 rounded-xl ${agent.color} text-white shadow-md min-w-[150px] cursor-default transition-shadow duration-200 hover:shadow-xl ${desktopGridPlacements[agent.id]}`}
              style={{ animation: "agentGlow 5s ease-in-out infinite", animationDelay: getAgentDelay(agent.id) }}
            >
              <span className="material-symbols-outlined text-[24px] transition-transform duration-200 group-hover/box:scale-110" style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider opacity-80">Agente {agent.id}</p>
                <p className="text-[13px] font-bold leading-tight">{agent.name.split("(")[0].trim().split("Agente de ").pop()}</p>
              </div>
            </div>
          ))}

          {/* Conector horizontal 1 -> 2 (Fila 1) */}
          <div className="col-start-2 row-start-1 relative flex items-center w-[52px] h-[44px]">
            <div
              className="absolute top-1/2 left-0 right-3 h-px bg-outline-variant/30 -translate-y-1/2"
              style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: "0.45s" }}
            />
            <span className="absolute right-0 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline-variant text-[14px]">chevron_right</span>
            <div
              className="absolute top-1/2 -translate-y-1/2 flow-dot rounded-full"
              style={{ animation: "dotHRight 5s ease-in-out infinite", animationDelay: "0.45s" }}
            />
          </div>

          {/* Conector horizontal 2 -> 3 (Fila 1) */}
          <div className="col-start-4 row-start-1 relative flex items-center w-[52px] h-[44px]">
            <div
              className="absolute top-1/2 left-0 right-3 h-px bg-outline-variant/30 -translate-y-1/2"
              style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: "1.35s" }}
            />
            <span className="absolute right-0 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline-variant text-[14px]">chevron_right</span>
            <div
              className="absolute top-1/2 -translate-y-1/2 flow-dot rounded-full"
              style={{ animation: "dotHRight 5s ease-in-out infinite", animationDelay: "1.35s" }}
            />
          </div>

          {/* Conector vertical 3 -> 4 (Caída derecha) */}
          <div className="col-start-5 row-start-2 relative flex flex-col items-center h-[32px] w-[12px]">
            <div
              className="absolute left-1/2 -translate-x-1/2 w-px h-full bg-outline-variant/30"
              style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: "2.25s" }}
            />
            <span className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 material-symbols-outlined text-outline-variant text-[14px]">expand_more</span>
            <div
              className="absolute left-1/2 -translate-x-1/2 flow-dot rounded-full"
              style={{ animation: "dotVDown 5s ease-in-out infinite", animationDelay: "2.25s" }}
            />
          </div>

          {/* Conector horizontal 4 -> 5 (Fila 2 - Flujo izquierda) */}
          <div className="col-start-4 row-start-3 relative flex items-center w-[52px] h-[44px]">
            <div
              className="absolute top-1/2 left-3 right-0 h-px bg-outline-variant/30 -translate-y-1/2"
              style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: "3.15s" }}
            />
            <span className="absolute left-0 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline-variant text-[14px]">chevron_left</span>
            <div
              className="absolute top-1/2 -translate-y-1/2 flow-dot rounded-full"
              style={{ animation: "dotHLeft 5s ease-in-out infinite", animationDelay: "3.15s" }}
            />
          </div>

          {/* Conector horizontal 5 -> 6 (Fila 2 - Flujo izquierda) */}
          <div className="col-start-2 row-start-3 relative flex items-center w-[52px] h-[44px]">
            <div
              className="absolute top-1/2 left-3 right-0 h-px bg-outline-variant/30 -translate-y-1/2"
              style={{ animation: `lineFlash 5s ease-in-out infinite`, animationDelay: "4.05s" }}
            />
            <span className="absolute left-0 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline-variant text-[14px]">chevron_left</span>
            <div
              className="absolute top-1/2 -translate-y-1/2 flow-dot rounded-full"
              style={{ animation: "dotHLeft 5s ease-in-out infinite", animationDelay: "4.05s" }}
            />
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-lg">
        {techStack.map((stack, i) => (
          <div
            key={stack.category}
            className="custom-card rounded-xl p-lg animate-fade-in-up"
            style={{ animationDelay: `${i * 75}ms` }}
          >
            <h4 className="text-label-md font-bold text-primary uppercase tracking-wider mb-md flex items-center gap-xs">
              <span className="w-2 h-2 rounded-full bg-primary inline-block" />
              {stack.category}
            </h4>
            <div className="space-y-sm">
              {stack.items.map((item) => (
                <div
                  key={item}
                  className="flex items-center gap-sm group/item p-xs rounded-lg hover:bg-surface-container transition-colors duration-150 cursor-default"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary transition-transform duration-200 group-hover/item:scale-150 flex-shrink-0" />
                  <span className="text-[13px] text-on-surface">{item}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Credits */}
      <div className="rounded-xl p-lg bg-primary-container border border-outline-variant shadow-[0px_4px_20px_rgba(30,58,95,0.04)]"
        style={{ color: "rgb(var(--color-on-primary-container))" }}>
        <div className="flex items-center gap-md mb-md">
          <span className="material-symbols-outlined text-[24px]" style={{ color: "rgb(var(--color-secondary-fixed))" }}>copyright</span>
          <h4 className="text-label-md font-bold uppercase tracking-wider">Créditos y Fuentes de Datos</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-lg text-[13px] opacity-90">
          <div>
            <p className="font-bold mb-xs" style={{ color: "rgb(var(--color-secondary-fixed))" }}>Datos Epidemiológicos</p>
            <p>OpenDengue Project — Datos abiertos de vigilancia de dengue a escala subnacional en América Latina.</p>
          </div>
          <div>
            <p className="font-bold mb-xs" style={{ color: "rgb(var(--color-secondary-fixed))" }}>Datos Climáticos</p>
            <p>NASA POWER API (NASA Langley Research Center) — Temperatura máxima/mínima, precipitación y humedad relativa histórica a nivel subnacional.</p>
          </div>
          <div>
            <p className="font-bold mb-xs" style={{ color: "rgb(var(--color-secondary-fixed))" }}>Datos Demográficos</p>
            <p>World Bank Open Data — Estimaciones poblacionales anuales por país y subregión.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
