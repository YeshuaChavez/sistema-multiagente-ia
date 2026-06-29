import React, { useState } from "react";

const viewSubtabs = {
  dashboard: [],
  predictor: ["Simulación", "Histórico", "Alertas"],
  explain: ["Global SHAP", "Local SHAP"],
  info: [],
};

const viewTitles = {
  dashboard: "Dashboard de Vigilancia",
  predictor: "Predictor en Vivo",
  explain: "Análisis de Explicabilidad (XAI)",
  info: "Información Técnica",
};

export default function Topbar({ currentView, activeSubtab, setActiveSubtab, onOpenSettings, darkMode, setDarkMode }) {
  const tabs = viewSubtabs[currentView] || [];
  const [showNotifs, setShowNotifs] = useState(false);

  const notifications = [
    { id: 1, text: "Backend conectado de forma exitosa.", time: "Hace 2 min", type: "success" },
    { id: 2, text: "Modelos XGBoost y LSTM PyTorch listos.", time: "Hace 5 min", type: "info" },
    { id: 3, text: "Dataset consolidado con 16,224 registros de entrenamiento (8 países, 2014–2022).", time: "Hace 10 min", type: "info" }
  ];

  return (
    <header className="sticky top-0 z-40 bg-surface border-b border-outline-variant">
      {/* ── Fila principal ── */}
      <div className="h-14 flex justify-between items-center w-full px-md sm:px-lg">
        <span className="text-[18px] font-bold text-primary truncate">
          {viewTitles[currentView] || "DenguePredict"}
        </span>

        <div className="flex items-center gap-xs sm:gap-sm relative">
          {/* Subtabs — solo en desktop md+ */}
          {tabs.length > 0 && (
            <div className="hidden md:flex items-center gap-lg mr-md">
              {tabs.map((tab) => (
                <span
                  key={tab}
                  onClick={() => setActiveSubtab(tab)}
                  className={`py-1 cursor-pointer transition-colors text-label-md whitespace-nowrap ${
                    activeSubtab === tab
                      ? "text-primary font-bold border-b-2 border-primary"
                      : "text-on-surface-variant hover:text-primary"
                  }`}
                >
                  {tab}
                </span>
              ))}
            </div>
          )}

          {/* Botones de acción */}
          <div className="flex items-center gap-xs text-on-surface-variant">
            {setDarkMode && (
              <button
                onClick={() => setDarkMode(!darkMode)}
                title={darkMode ? "Modo claro" : "Modo oscuro"}
                className="p-xs sm:p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: "'FILL' 1" }}>
                  {darkMode ? "light_mode" : "dark_mode"}
                </span>
              </button>
            )}
            <button
              onClick={() => setShowNotifs(!showNotifs)}
              className="p-xs sm:p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer relative"
            >
              <span className="material-symbols-outlined text-[20px]">notifications</span>
              <span className="absolute top-1 right-1 w-2 h-2 bg-secondary rounded-full border-2 border-surface"></span>
            </button>
            <button
              onClick={onOpenSettings}
              className="p-xs sm:p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer"
            >
              <span className="material-symbols-outlined text-[20px]">settings</span>
            </button>
          </div>

          {/* Dropdown de notificaciones */}
          {showNotifs && (
            <div className="absolute right-0 top-12 bg-surface-container-low border border-outline-variant rounded-xl shadow-xl w-72 p-md flex flex-col gap-sm z-50 animate-fade-in">
              <div className="flex justify-between items-center border-b border-outline-variant pb-xs">
                <span className="text-[12px] font-bold text-primary uppercase">Notificaciones ({notifications.length})</span>
                <span className="text-[10px] text-on-surface-variant hover:text-primary cursor-pointer" onClick={() => setShowNotifs(false)}>Marcar leído</span>
              </div>
              <div className="flex flex-col gap-sm">
                {notifications.map((n) => (
                  <div key={n.id} className="flex gap-sm items-start text-[11px] leading-relaxed p-xs hover:bg-surface-container rounded-lg transition-colors">
                    <span className={`material-symbols-outlined text-[16px] mt-0.5 ${n.type === "success" ? "text-emerald-500" : "text-sky-500"}`}>
                      {n.type === "success" ? "check_circle" : "info"}
                    </span>
                    <div>
                      <p className="text-on-background">{n.text}</p>
                      <span className="text-[9px] text-on-surface-variant">{n.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Fila de subtabs en móvil ── */}
      {tabs.length > 0 && (
        <div className="md:hidden flex overflow-x-auto border-t border-outline-variant/40 px-md">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveSubtab(tab)}
              className={`flex-shrink-0 px-md py-2 text-label-md transition-colors whitespace-nowrap border-b-2 ${
                activeSubtab === tab
                  ? "text-primary font-bold border-primary"
                  : "text-on-surface-variant border-transparent hover:text-primary"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      )}
    </header>
  );
}
