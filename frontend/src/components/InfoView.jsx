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
      "Combina las predicciones de los Agentes 3 y 4 usando pesos ajustados dinámicamente por el Agente 6. En régimen normal usa pesos proporcionales al R² (w_XGB=0.512, w_LSTM=0.488). Clasifica la predicción final en niveles de riesgo calibrados con percentiles históricos por departamento.",
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
              className="custom-card rounded-xl p-lg animate-fade-in-up group"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="flex items-start gap-lg">
                {/* Agent Icon */}
                <div className={`w-14 h-14 rounded-xl ${agent.color} flex items-center justify-center text-white flex-shrink-0 shadow-md transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}>
                  <span className="material-symbols-outlined text-[28px]" style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
                </div>

                <div className="flex-1">
                  {/* Header */}
                  <div className="flex items-center gap-sm mb-sm">
                    <span className="text-[10px] font-bold text-white px-2 py-0.5 rounded-full bg-primary-container">
                      AGENTE {agent.id}
                    </span>
                    <h4 className="text-[15px] font-bold text-primary">{agent.name}</h4>
                  </div>

                  {/* Description */}
                  <p className="text-[13px] text-on-surface-variant leading-relaxed mb-md">
                    {agent.description}
                  </p>

                  {/* I/O */}
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

                  {/* Tech Tags */}
                  <div className="flex flex-wrap gap-xs">
                    {agent.tech.map((t) => (
                      <span
                        key={t}
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${agent.colorLight} text-on-surface-variant
                          transition-all duration-150 hover:scale-105 hover:shadow-sm cursor-default`}
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Flow */}
      <div className="custom-card rounded-xl p-lg">
        <h3 className="text-label-md font-bold text-primary uppercase tracking-wider mb-lg flex items-center gap-sm">
          <span className="material-symbols-outlined text-[18px]">account_tree</span>
          Flujo de la Arquitectura
        </h3>
        <div className="flex flex-col items-center gap-md">
          {[agents.slice(0, 3), agents.slice(3, 6)].map((row, rowIdx) => (
            <React.Fragment key={rowIdx}>
              <div className="flex items-center gap-sm">
                {row.map((agent, idx) => (
                  <React.Fragment key={agent.id}>
                    <div className={`group/box flex items-center gap-sm px-5 py-3 rounded-xl ${agent.color} text-white shadow-md min-w-[130px] cursor-default
                      transition-all duration-200 hover:-translate-y-1 hover:shadow-xl`}>
                      <span className="material-symbols-outlined text-[24px] transition-transform duration-200 group-hover/box:scale-110"
                        style={{ fontVariationSettings: "'FILL' 1" }}>{agent.icon}</span>
                      <div>
                        <p className="text-[10px] font-bold uppercase tracking-wider opacity-80">Agente {agent.id}</p>
                        <p className="text-[13px] font-bold leading-tight">{agent.name.split("(")[0].trim().split("Agente de ").pop()}</p>
                      </div>
                    </div>
                    {idx < 2 && (
                      <span className="material-symbols-outlined text-outline text-[22px]">arrow_forward</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
              {rowIdx === 0 && (
                <span className="material-symbols-outlined text-outline text-[22px]">arrow_downward</span>
              )}
            </React.Fragment>
          ))}
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
      <div className="rounded-xl p-lg bg-primary-container text-on-primary border border-outline-variant shadow-[0px_4px_20px_rgba(30,58,95,0.04)]">
        <div className="flex items-center gap-md mb-md">
          <span className="material-symbols-outlined text-secondary-fixed text-[24px]">copyright</span>
          <h4 className="text-label-md font-bold uppercase tracking-wider">Créditos y Fuentes de Datos</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-lg text-[13px] opacity-90">
          <div>
            <p className="font-bold text-secondary-fixed mb-xs">Datos Epidemiológicos</p>
            <p>OpenDengue Project — Datos abiertos de vigilancia de dengue a escala subnacional en América Latina.</p>
          </div>
          <div>
            <p className="font-bold text-secondary-fixed mb-xs">Datos Climáticos</p>
            <p>NASA POWER API (NASA Langley Research Center) — Temperatura máxima/mínima, precipitación y humedad relativa histórica a nivel subnacional.</p>
          </div>
          <div>
            <p className="font-bold text-secondary-fixed mb-xs">Datos Demográficos</p>
            <p>World Bank Open Data — Estimaciones poblacionales anuales por país y subregión.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
