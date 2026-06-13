import React, { useState } from "react";

const viewSubtabs = {
  dashboard: [],
  predictor: ["Simulación", "Histórico", "Alertas"],
  explain: ["Global SHAP", "Local SHAP"],
  info: [],
};

export default function Topbar({ currentView, activeSubtab, setActiveSubtab, onOpenSettings, darkMode, setDarkMode }) {
  const tabs = viewSubtabs[currentView] || [];
  const [showNotifs, setShowNotifs] = useState(false);

  const notifications = [
    { id: 1, text: "Backend conectado de forma exitosa.", time: "Hace 2 min", type: "success" },
    { id: 2, text: "Modelos LightGBM y LSTM PyTorch listos.", time: "Hace 5 min", type: "info" },
    { id: 3, text: "Dataset consolidado con 13,585 registros.", time: "Hace 10 min", type: "info" }
  ];

  return (
    <header className="h-16 flex justify-between items-center w-full px-lg bg-surface border-b border-outline-variant sticky top-0 z-40">
      <div className="flex items-center gap-md">
        <span className="text-headline-md font-bold text-primary">EpiPredict Dengue</span>
      </div>

      <div className="flex items-center gap-md relative">
        {/* View Tabs */}
        {tabs.length > 0 && (
          <div className="hidden md:flex items-center gap-lg mr-md">
            {tabs.map((tab) => {
              const isTabActive = activeSubtab === tab;
              return (
                <span
                  key={tab}
                  onClick={() => setActiveSubtab(tab)}
                  className={`py-1 cursor-pointer transition-colors text-label-md ${
                    isTabActive
                      ? "text-primary font-bold border-b-2 border-primary"
                      : "text-on-surface-variant hover:text-primary"
                  }`}
                >
                  {tab}
                </span>
              );
            })}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-sm text-on-surface-variant">
          {setDarkMode && (
            <button
              onClick={() => setDarkMode(!darkMode)}
              title={darkMode ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
              className="p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer"
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
                {darkMode ? "light_mode" : "dark_mode"}
              </span>
            </button>
          )}
          <button
            onClick={() => setShowNotifs(!showNotifs)}
            className="p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer relative"
          >
            <span className="material-symbols-outlined">notifications</span>
            <span className="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-secondary rounded-full border-2 border-surface"></span>
          </button>
          <button
            onClick={onOpenSettings}
            className="p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer"
          >
            <span className="material-symbols-outlined">settings</span>
          </button>
          <div className="w-8 h-8 rounded-full bg-primary-fixed flex items-center justify-center ml-sm">
            <span className="material-symbols-outlined text-primary text-[20px]">account_circle</span>
          </div>
        </div>

        {/* NOTIFICATIONS DROPDOWN */}
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
    </header>
  );
}
