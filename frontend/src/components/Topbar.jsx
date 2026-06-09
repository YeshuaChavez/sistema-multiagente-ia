import React from "react";

const viewSubtabs = {
  dashboard: [],
  predictor: ["Simulación", "Histórico", "Alertas"],
  explain: ["Global SHAP", "Local SHAP"],
  info: [],
};

export default function Topbar({ currentView }) {
  const tabs = viewSubtabs[currentView] || [];

  return (
    <header className="h-16 flex justify-between items-center w-full px-lg bg-surface border-b border-outline-variant sticky top-0 z-40">
      <div className="flex items-center gap-md">
        <span className="text-headline-md font-bold text-primary">EpiPredict Dengue</span>
      </div>

      <div className="flex items-center gap-md">
        {/* View Tabs */}
        {tabs.length > 0 && (
          <div className="hidden md:flex items-center gap-lg mr-md">
            {tabs.map((tab, idx) => (
              <span
                key={tab}
                className={`py-1 cursor-pointer transition-colors text-label-md ${
                  idx === 0
                    ? "text-primary font-bold border-b-2 border-primary"
                    : "text-on-surface-variant hover:text-primary"
                }`}
              >
                {tab}
              </span>
            ))}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-sm text-on-surface-variant">
          <button className="p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer">
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button className="p-sm hover:bg-surface-container rounded-full transition-colors cursor-pointer">
            <span className="material-symbols-outlined">settings</span>
          </button>
          <div className="w-8 h-8 rounded-full bg-primary-fixed flex items-center justify-center ml-sm">
            <span className="material-symbols-outlined text-primary text-[20px]">account_circle</span>
          </div>
        </div>
      </div>
    </header>
  );
}
