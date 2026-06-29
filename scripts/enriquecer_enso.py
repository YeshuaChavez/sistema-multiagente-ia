# -*- coding: utf-8 -*-
"""
Enriquece dataset_features_latam.csv con índices ENSO continuos de NOAA:
  - nino34_index  : Niño 3.4 anomaly (estándar global)
  - nino12_index  : Niño 1+2 anomaly (más relevante para costa peruana/ecuatoriana)
  - nino34_lag1/2/3 : rezagos de Niño 3.4
  - nino12_lag1/2/3 : rezagos de Niño 1+2

Reemplaza los binarios indicador_nino / indicador_nina por valores continuos.
Los binarios originales se mantienen para retrocompatibilidad.

Fuente: NOAA CPC — https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices
"""

import os
import io
import requests
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN   = os.path.join(BASE, "data", "processed", "dataset_features_latam.csv")
OUT  = os.path.join(BASE, "data", "processed", "dataset_features_latam_enso.csv")

# ── 1. Descargar índices SST de NOAA ──────────────────────────────────────────
print("[ENSO] Descargando índices NOAA CPC...")
URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
resp = requests.get(URL, timeout=30)
resp.raise_for_status()

# Formato: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
lines = [l for l in resp.text.strip().splitlines() if l.strip() and not l.startswith("YR")]
rows = []
for line in lines:
    parts = line.split()
    if len(parts) >= 10:
        rows.append({
            "ano": int(parts[0]),
            "mes": int(parts[1]),
            "nino12_raw":  float(parts[2]),
            "nino12_index": float(parts[3]),   # anomalía 1+2
            "nino34_raw":  float(parts[8]),
            "nino34_index": float(parts[9]),   # anomalía 3.4
        })

df_enso = pd.DataFrame(rows)
print(f"[ENSO] Descargados {len(df_enso)} registros ({df_enso.ano.min()}–{df_enso.ano.max()})")

# ── 2. Agregar rezagos del índice ENSO ────────────────────────────────────────
df_enso = df_enso.sort_values(["ano","mes"]).reset_index(drop=True)
for lag in [1, 2, 3]:
    df_enso[f"nino34_lag{lag}"] = df_enso["nino34_index"].shift(lag)
    df_enso[f"nino12_lag{lag}"] = df_enso["nino12_index"].shift(lag)

# ── 3. Leer dataset principal ─────────────────────────────────────────────────
print("[ENSO] Leyendo dataset principal...")
df = pd.read_csv(IN)
print(f"[ENSO] Dataset: {df.shape}")

# ── 4. Merge por año-mes ──────────────────────────────────────────────────────
df = df.merge(
    df_enso[["ano","mes","nino34_index","nino12_index",
             "nino34_lag1","nino34_lag2","nino34_lag3",
             "nino12_lag1","nino12_lag2","nino12_lag3"]],
    on=["ano","mes"],
    how="left"
)

missing = df["nino34_index"].isna().sum()
print(f"[ENSO] Registros sin índice ENSO: {missing} (se rellenan con 0)")
for col in ["nino34_index","nino12_index",
            "nino34_lag1","nino34_lag2","nino34_lag3",
            "nino12_lag1","nino12_lag2","nino12_lag3"]:
    df[col] = df[col].fillna(0.0)

# ── 5. Guardar ────────────────────────────────────────────────────────────────
df.to_csv(OUT, index=False)
print(f"[ENSO] Guardado en: {OUT}")
print(f"[ENSO] Shape final: {df.shape}")
print(f"\n[ENSO] Nuevas columnas agregadas:")
new_cols = ["nino34_index","nino12_index",
            "nino34_lag1","nino34_lag2","nino34_lag3",
            "nino12_lag1","nino12_lag2","nino12_lag3"]
for c in new_cols:
    print(f"  {c:25s}  min={df[c].min():.2f}  max={df[c].max():.2f}  mean={df[c].mean():.2f}")

print("\n[ENSO] Listo. Usa dataset_features_latam_enso.csv para reentrenar.")
print("[ENSO] En el Agente 2 actualiza COLS_EXCLUIR para incluir indicador_nino/nina")
print("[ENSO] y agregar las nuevas columnas nino34_index y nino12_index como features.")
