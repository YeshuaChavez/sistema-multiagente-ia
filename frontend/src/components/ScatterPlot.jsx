import { useState, useMemo } from "react";

const ISO_COLORS = {
  PER: "#ef4444", BRA: "#22c55e", COL: "#3b82f6", MEX: "#f59e0b",
  ARG: "#8b5cf6", BOL: "#ec4899", ECU: "#14b8a6", PAN: "#f97316",
};
const ISO_NAMES = {
  PER: "Perú", BRA: "Brasil", COL: "Colombia", MEX: "México",
  ARG: "Argentina", BOL: "Bolivia", ECU: "Ecuador", PAN: "Panamá",
};

const W = 520, H = 340;
const PAD = { top: 20, right: 24, bottom: 48, left: 56 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

function niceMax(v) {
  const steps = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000];
  return steps.find((s) => s >= v) ?? Math.ceil(v / 200) * 200;
}

export default function ScatterPlot({ data, darkMode, metrics }) {
  const c = {
    grid:    darkMode ? "#2a3a50" : "#e2e8f0",
    tick:    darkMode ? "#8faabb" : "#94a3b8",
    axis:    darkMode ? "#8faabb" : "#334155",
    label:   darkMode ? "#aec8d8" : "#475569",
    tipBg:   darkMode ? "#1a2737" : "white",
    tipBord: darkMode ? "#2a3a50" : "#cbd5e1",
    tipHead: darkMode ? "#adc8e5" : "#1e3a5f",
    tipText: darkMode ? "#8faabb" : "#475569",
  };
  const [model, setModel] = useState("ensemble");
  const [hovered, setHovered] = useState(null);

  const points = data?.[model] ?? [];
  const rawMeta = data?.metricas?.[model];
  const meta = (() => {
    const apiMeta = {
      ensemble: metrics?.r2_ensemble != null ? { r2: metrics.r2_ensemble, mae: metrics.mae_ensemble?.toFixed(2), rmse: metrics.rmse_ensemble?.toFixed(2) } : null,
      xgb:      metrics?.r2_xgb      != null ? { r2: metrics.r2_xgb,      mae: metrics.mae_xgb?.toFixed(2),      rmse: metrics.rmse_xgb?.toFixed(2)      } : null,
      lstm:     metrics?.r2_lstm     != null ? { r2: metrics.r2_lstm,     mae: metrics.mae_lstm?.toFixed(2),     rmse: metrics.rmse_lstm?.toFixed(2)     } : null,
    };
    return apiMeta[model] ?? rawMeta;
  })();

  const maxVal = useMemo(() => {
    if (!points.length) return 200;
    return niceMax(Math.max(...points.map((p) => Math.max(p.actual, p.pred))));
  }, [points]);

  const sx = (v) => PAD.left + (v / maxVal) * PW;
  const sy = (v) => PAD.top + PH - (v / maxVal) * PH;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * maxVal));

  const MODELS = [
    { id: "ensemble", label: "Ensemble" },
    { id: "xgboost",  label: "XGBoost" },
    { id: "lstm",     label: "LSTM" },
  ];

  if (!data) {
    return (
      <div className="custom-card rounded-xl p-lg flex items-center justify-center min-h-[200px]">
        <div className="text-center text-on-surface-variant">
          <span className="material-symbols-outlined text-[40px] opacity-40 block mb-sm">scatter_plot</span>
          <p className="text-[13px]">Generando datos del diagrama...</p>
          <p className="text-[11px] opacity-60 mt-xs">Ejecuta <code className="bg-surface-container px-xs rounded">generar_scatter_data.py</code> para habilitar esta vista</p>
        </div>
      </div>
    );
  }

  return (
    <div className="custom-card rounded-xl p-lg">
      {/* Header */}
      <div className="flex items-start justify-between mb-md flex-wrap gap-sm">
        <div>
          <h4 className="text-label-md font-bold text-primary uppercase tracking-wider flex items-center gap-sm">
            <span className="material-symbols-outlined text-[18px]">scatter_plot</span>
            Diagrama de Dispersión — Real vs Predicho
          </h4>
          <p className="text-[12px] text-on-surface-variant mt-xs">
            Set de prueba 2021–2022 · {points.length.toLocaleString()} observaciones completas
            {meta && (
              <>
                {" · "}
                <strong className="text-primary">R²={( meta.r2 * 100).toFixed(2)}%</strong>
                {" · MAE="}
                <strong className="text-primary">{meta.mae}</strong>
                {" · RMSE="}
                <strong className="text-primary">{meta.rmse}</strong>
              </>
            )}
          </p>
        </div>
        <div className="flex gap-xs">
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => setModel(m.id)}
              className={`text-[11px] font-bold px-sm py-1 rounded-lg border transition-colors cursor-pointer
                ${model === m.id
                  ? "bg-primary text-on-primary border-primary"
                  : "border-outline-variant text-on-surface-variant hover:bg-surface-container"}`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Chart */}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 360 }}
        onMouseLeave={() => setHovered(null)}>

        {/* Grid */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={PAD.top} x2={sx(t)} y2={PAD.top + PH}
              stroke={c.grid} strokeWidth="1" />
            <line x1={PAD.left} y1={sy(t)} x2={PAD.left + PW} y2={sy(t)}
              stroke={c.grid} strokeWidth="1" />
            <text x={sx(t)} y={H - PAD.bottom + 16} textAnchor="middle"
              fontSize="10" fill={c.tick}>{t}</text>
            {t > 0 && (
              <text x={PAD.left - 8} y={sy(t) + 4} textAnchor="end"
                fontSize="10" fill={c.tick}>{t}</text>
            )}
          </g>
        ))}

        {/* Perfect prediction diagonal */}
        <line x1={sx(0)} y1={sy(0)} x2={sx(maxVal)} y2={sy(maxVal)}
          stroke={c.tick} strokeWidth="1.5" strokeDasharray="6,4" />

        {/* Points */}
        {points.map((p, i) => (
          <circle key={i}
            cx={sx(p.actual)} cy={sy(p.pred)}
            r="3" fill={ISO_COLORS[p.iso] ?? "#6366f1"} opacity="0.45"
            className="cursor-pointer"
            onMouseEnter={(e) => setHovered({ ...p, mx: sx(p.actual), my: sy(p.pred) })}
          />
        ))}

        {/* Tooltip */}
        {hovered && (() => {
          const tipX = hovered.mx > W * 0.65 ? hovered.mx - 122 : hovered.mx + 8;
          const tipY = hovered.my < 50 ? hovered.my + 6 : hovered.my - 50;
          return (
            <g>
              <rect x={tipX} y={tipY} width="116" height="44" rx="5"
                fill={c.tipBg} stroke={c.tipBord} strokeWidth="1" filter="drop-shadow(0 2px 4px rgba(0,0,0,.18))" />
              <text x={tipX + 7} y={tipY + 15} fontSize="10.5" fill={c.tipHead} fontWeight="700">
                {ISO_NAMES[hovered.iso] ?? hovered.iso} · {hovered.ano}
              </text>
              <text x={tipX + 7} y={tipY + 29} fontSize="10" fill={c.tipText}>
                Real: <tspan fontWeight="600">{hovered.actual}</tspan>
              </text>
              <text x={tipX + 7} y={tipY + 41} fontSize="10" fill={c.tipText}>
                Pred: <tspan fontWeight="600">{hovered.pred}</tspan>
              </text>
            </g>
          );
        })()}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + PH}
          stroke={c.axis} strokeWidth="1.5" />
        <line x1={PAD.left} y1={PAD.top + PH} x2={PAD.left + PW} y2={PAD.top + PH}
          stroke={c.axis} strokeWidth="1.5" />

        {/* Axis labels */}
        <text x={PAD.left + PW / 2} y={H - 4} textAnchor="middle"
          fontSize="11" fill={c.label} fontWeight="600">
          Incidencia real (casos/100k hab.)
        </text>
        <text
          transform={`translate(13,${PAD.top + PH / 2}) rotate(-90)`}
          textAnchor="middle" fontSize="11" fill={c.label} fontWeight="600">
          Predicción (casos/100k hab.)
        </text>
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-md mt-sm pt-sm border-t border-outline-variant/30">
        {Object.entries(ISO_COLORS).map(([iso, color]) => (
          <span key={iso} className="flex items-center gap-xs text-[11px] text-on-surface-variant">
            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
            {ISO_NAMES[iso] ?? iso}
          </span>
        ))}
        <span className="flex items-center gap-xs text-[11px] text-on-surface-variant ml-auto">
          <svg width="22" height="10">
            <line x1="0" y1="5" x2="22" y2="5" stroke={c.tick} strokeWidth="1.5" strokeDasharray="5,3" />
          </svg>
          Predicción perfecta (y=x)
        </span>
      </div>
    </div>
  );
}
