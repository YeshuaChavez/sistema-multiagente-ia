import React from "react";

const navItems = [
  { id: "dashboard", label: "Dashboard",  icon: "dashboard" },
  { id: "predictor", label: "Predictor",  icon: "query_stats" },
  { id: "explain",   label: "XAI",        icon: "psychology" },
  { id: "info",      label: "Info",        icon: "menu_book" },
];

export default function BottomNav({ currentView, setCurrentView }) {
  return (
    <nav
      className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-outline-variant flex items-stretch"
      style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)", minHeight: "4rem" }}
    >
      {navItems.map((item) => {
        const isActive = currentView === item.id;
        return (
          <button
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            className={`relative flex-1 flex flex-col items-center justify-center gap-[3px] transition-colors duration-150 cursor-pointer
              ${isActive ? "text-primary" : "text-on-surface-variant"}`}
          >
            {isActive && (
              <span className="absolute top-0 left-1/2 -translate-x-1/2 w-10 h-[3px] bg-primary rounded-b-full" />
            )}
            <span
              className={`material-symbols-outlined text-[22px] transition-transform duration-150 ${isActive ? "scale-110" : ""}`}
              style={{ fontVariationSettings: isActive ? "'FILL' 1, 'wght' 600" : "'FILL' 0, 'wght' 400" }}
            >
              {item.icon}
            </span>
            <span className={`text-[10px] leading-none ${isActive ? "font-bold" : "font-semibold"}`}>
              {item.label}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
