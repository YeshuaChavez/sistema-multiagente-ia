import React from "react";

const menuItems = [
  { id: "dashboard", label: "Dashboard", icon: "dashboard" },
  { id: "predictor", label: "Predictor en Vivo", icon: "query_stats" },
  { id: "explain", label: "XAI", icon: "psychology" },
  { id: "info", label: "Info Técnica", icon: "menu_book" },
];

export default function Sidebar({ currentView, setCurrentView, onOpenSettings, onOpenSupport, onGenerateReport }) {

  return (
    <aside className="h-screen w-64 fixed left-0 top-0 bg-surface-container-low border-r border-outline-variant flex flex-col py-lg z-50 hidden md:flex">
      {/* Logo */}
      <div className="px-md mb-xl">
        <h1 className="font-black text-headline-md text-primary">DenguePredict</h1>
        <p className="text-label-md text-on-surface-variant">Vigilancia y Alerta Temprana</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-xs">
        {menuItems.map((item) => {
          const isActive = currentView === item.id;
          return (
            <div
              key={item.id}
              onClick={() => setCurrentView(item.id)}
              className={`flex items-center gap-md px-md py-sm rounded-lg mx-sm mb-xs cursor-pointer transition-colors ${
                isActive
                  ? "text-primary font-bold bg-surface-container-high"
                  : "text-on-surface-variant hover:bg-surface-container-highest"
              }`}
            >
              <span className="material-symbols-outlined" style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}>
                {item.icon}
              </span>
              <span className="text-label-md">{item.label}</span>
            </div>
          );
        })}
      </nav>

      {/* Bottom Actions */}
      <div className="mt-auto pt-lg border-t border-outline-variant/30 space-y-xs">
        <button 
          onClick={onGenerateReport}
          className="mx-md mb-md bg-primary text-on-primary px-md py-sm rounded-lg text-label-md font-bold hover:opacity-90 transition-all flex items-center justify-center gap-sm w-[calc(100%-32px)] cursor-pointer"
        >
          <span className="material-symbols-outlined text-[18px]">summarize</span>
          Generar Reporte
        </button>
        <div 
          onClick={onOpenSettings}
          className="flex items-center gap-md px-md py-sm rounded-lg mx-sm cursor-pointer hover:bg-surface-container-highest transition-colors text-on-surface-variant"
        >
          <span className="material-symbols-outlined">settings</span>
          <span className="text-label-md">Ajustes</span>
        </div>
        <div 
          onClick={onOpenSupport}
          className="flex items-center gap-md px-md py-sm rounded-lg mx-sm cursor-pointer hover:bg-surface-container-highest transition-colors text-on-surface-variant"
        >
          <span className="material-symbols-outlined">contact_support</span>
          <span className="text-label-md">Soporte</span>
        </div>
      </div>
    </aside>
  );
}
