import React from "react";
import MosquitoIcon from "./MosquitoIcon";

const menuItems = [
  { id: "dashboard", label: "Dashboard",        icon: "dashboard" },
  { id: "predictor", label: "Predictor en Vivo", icon: "query_stats" },
  { id: "explain",   label: "XAI",               icon: "psychology" },
  { id: "info",      label: "Info Técnica",       icon: "menu_book" },
];

export default function Sidebar({ currentView, setCurrentView, onOpenSettings, onOpenSupport, onGenerateReport }) {
  return (
    <aside className="h-screen w-64 fixed left-0 top-0 bg-surface-container-low border-r border-outline-variant flex flex-col py-lg z-50 hidden md:flex">

      {/* Logo */}
      <div className="px-md mb-xl">
        <div className="flex items-center gap-sm mb-xs">
          <MosquitoIcon size={28} className="text-primary animate-float flex-shrink-0" />
          <h1 className="font-black text-headline-md text-primary">DenguePredict</h1>
        </div>
        <p className="text-label-md text-on-surface-variant pl-[36px]">Vigilancia y Alerta Temprana</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-xs px-sm">
        {menuItems.map((item, i) => {
          const isActive = currentView === item.id;
          return (
            <div
              key={item.id}
              onClick={() => setCurrentView(item.id)}
              className={`
                group relative flex items-center gap-md px-md py-sm rounded-xl cursor-pointer
                transition-all duration-200 select-none
                ${isActive
                  ? "bg-primary text-on-primary shadow-md"
                  : "text-on-surface-variant hover:bg-surface-container-highest hover:text-on-surface"}
              `}
              style={{ animationDelay: `${i * 50}ms` }}
            >
              {/* Active left bar */}
              {isActive && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-[55%] bg-on-primary rounded-r-full opacity-60" />
              )}

              <span
                className="material-symbols-outlined text-[22px] transition-transform duration-200 group-hover:scale-110"
                style={{
                  fontVariationSettings: isActive ? "'FILL' 1, 'wght' 600" : "'FILL' 0, 'wght' 400",
                }}
              >
                {item.icon}
              </span>
              <span className="text-label-md font-semibold">{item.label}</span>

              {/* Hover arrow */}
              {!isActive && (
                <span className="material-symbols-outlined text-[14px] ml-auto opacity-0 group-hover:opacity-40 transition-opacity duration-150 -translate-x-1 group-hover:translate-x-0 transition-transform">
                  chevron_right
                </span>
              )}
            </div>
          );
        })}
      </nav>

      {/* Bottom Actions */}
      <div className="mt-auto pt-lg border-t border-outline-variant/30 space-y-xs px-sm">
        <button
          onClick={onGenerateReport}
          className="btn-primary w-full bg-primary text-on-primary px-md py-sm rounded-xl text-label-md font-bold
            hover:opacity-90 flex items-center justify-center gap-sm cursor-pointer shadow-md
            hover:shadow-lg transition-all duration-200 group"
        >
          <span className="material-symbols-outlined text-[18px] transition-transform duration-200 group-hover:scale-110"
            style={{ fontVariationSettings: "'FILL' 1" }}>
            summarize
          </span>
          Generar Reporte
        </button>

        {[
          { icon: "settings",        label: "Ajustes", onClick: onOpenSettings },
          { icon: "contact_support", label: "Soporte",  onClick: onOpenSupport  },
        ].map(({ icon, label, onClick }) => (
          <div
            key={label}
            onClick={onClick}
            className="group flex items-center gap-md px-md py-sm rounded-xl cursor-pointer
              hover:bg-surface-container-highest transition-all duration-200 text-on-surface-variant hover:text-on-surface"
          >
            <span className="material-symbols-outlined text-[20px] transition-transform duration-200 group-hover:scale-110">
              {icon}
            </span>
            <span className="text-label-md">{label}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}
