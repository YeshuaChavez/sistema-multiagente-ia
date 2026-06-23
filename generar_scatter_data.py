# -*- coding: utf-8 -*-
"""
Genera scatter_data.json con predicciones vs reales del set de prueba (2021-2022).
Ejecutar UNA vez localmente. Sube el resultado a S3.

Uso:
    python generar_scatter_data.py
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
load_dotenv('.env')
sys.path.insert(0, 'agentes')

import s3_client as s3

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DB      = os.path.join(BASE, "Base de Datos")
MODELS  = os.path.join(DB, "modelos")
PROC    = os.path.join(DB, "datos_procesados")

feat_path   = os.path.join(PROC, "dataset_features_latam.csv")
master_path = os.path.join(PROC, "dataset_maestro_mensual_latam.csv")

# ── Cargar features ────────────────────────────────────────────────────────────
print("[1/5] Cargando dataset de features...")
df = pd.read_csv(feat_path)
df_master = pd.read_csv(master_path)

# Split test: años 2021-2022
mask_test = df['ano'].isin([2021, 2022])
df_test   = df[mask_test].copy()
print(f"      Test set: {len(df_test)} filas ({df_test['ano'].value_counts().to_dict()})")

# ── Cargar modelo XGBoost ──────────────────────────────────────────────────────
print("[2/5] Cargando modelos...")
with open(os.path.join(MODELS, "pipeline_ml.pkl"), "rb") as f:
    pipeline_xgb = pickle.load(f)
with open(os.path.join(MODELS, "cols_feat.pkl"), "rb") as f:
    cols_feat = pickle.load(f)
with open(os.path.join(MODELS, "metrics.json")) as f:
    metrics = json.load(f)

w_xgb  = metrics.get("ensemble_w_xgb",  0.512)
w_lstm = metrics.get("ensemble_w_lstm", 0.488)

# ── Cargar modelo LSTM ────────────────────────────────────────────────────────
import torch
import torch.nn as nn

with open(os.path.join(MODELS, "lstm_config.json")) as f:
    lstm_cfg = json.load(f)
with open(os.path.join(MODELS, "lstm_features.pkl"), "rb") as f:
    lstm_feats = pickle.load(f)
with open(os.path.join(MODELS, "escalador_lstm.pkl"), "rb") as f:
    scaler_lstm = pickle.load(f)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# Detectar num_layers desde el state_dict
_sd = torch.load(os.path.join(MODELS, "lstm_model.pth"), map_location="cpu", weights_only=True)
_num_layers = sum(1 for k in _sd if k.startswith("lstm.weight_ih_l"))

model_lstm = LSTMModel(
    lstm_cfg["input_dim"], lstm_cfg["hidden_dim"],
    _num_layers, lstm_cfg.get("dropout", 0.2)
)
model_lstm.load_state_dict(_sd)
model_lstm.eval()

SEQ_LEN = lstm_cfg.get("seq_len", 12)

# ── Predicciones XGBoost ───────────────────────────────────────────────────────
print("[3/5] Predicciones XGBoost...")
X_test = df_test[cols_feat].values
pred_xgb_log = pipeline_xgb.predict(X_test)
pred_xgb     = np.expm1(pred_xgb_log)

# ── Predicciones LSTM ─────────────────────────────────────────────────────────
print("[4/5] Predicciones LSTM...")
df_all = df.copy()
df_all_scaled = df_all.copy()
df_all_scaled[lstm_feats] = scaler_lstm.transform(df_all[lstm_feats])

all_indices  = list(df_test.index)
pred_lstm    = []
y_test_lstm  = []

for idx in all_indices:
    pos = df_all.index.get_loc(idx)
    if pos < SEQ_LEN:
        pred_lstm.append(np.nan)
        y_test_lstm.append(np.nan)
        continue
    seq = df_all_scaled.iloc[pos - SEQ_LEN: pos][lstm_feats].values
    x_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p_log = model_lstm(x_t).item()
    pred_lstm.append(max(0.0, np.expm1(p_log)))
    y_test_lstm.append(df_test.loc[idx, "incidencia_dengue"])

pred_lstm_arr = np.array(pred_lstm, dtype=float)

# ── Ensemble ───────────────────────────────────────────────────────────────────
valid  = ~np.isnan(pred_lstm_arr)
y_real = df_test["incidencia_dengue"].values
pred_ens = w_xgb * pred_xgb + w_lstm * np.where(valid, pred_lstm_arr, pred_xgb)

# ── Métricas ───────────────────────────────────────────────────────────────────
print("[5/5] Calculando métricas y guardando...")
r2_ens   = r2_score(np.log1p(y_real), np.log1p(pred_ens))
mae_ens  = mean_absolute_error(y_real, pred_ens)
rmse_ens = np.sqrt(mean_squared_error(y_real, pred_ens))
r2_xgb_v  = r2_score(np.log1p(y_real), np.log1p(pred_xgb))
mae_xgb_v  = mean_absolute_error(y_real, pred_xgb)
rmse_xgb_v = np.sqrt(mean_squared_error(y_real, pred_xgb))
r2_lstm_v  = r2_score(np.log1p(y_real[valid]), np.log1p(pred_lstm_arr[valid]))
mae_lstm_v  = mean_absolute_error(y_real[valid], pred_lstm_arr[valid])
rmse_lstm_v = np.sqrt(mean_squared_error(y_real[valid], pred_lstm_arr[valid]))

print(f"  Ensemble  — R²={r2_ens*100:.2f}%  MAE={mae_ens:.2f}  RMSE={rmse_ens:.2f}")
print(f"  XGBoost   — R²={r2_xgb_v*100:.2f}%  MAE={mae_xgb_v:.2f}  RMSE={rmse_xgb_v:.2f}")
print(f"  LSTM      — R²={r2_lstm_v*100:.2f}%  MAE={mae_lstm_v:.2f}  RMSE={rmse_lstm_v:.2f}")

# ── Actualizar metrics.json ───────────────────────────────────────────────────
metrics.update({
    "rmse_xgb":      round(rmse_xgb_v, 4),
    "rmse_lstm":     round(rmse_lstm_v, 4),
    "rmse_ensemble": round(rmse_ens, 4),
    "n_train":       int((~mask_test).sum()),
    "n_test":        int(mask_test.sum()),
    "n_paises":      int(df["iso_a0"].nunique()) if "iso_a0" in df.columns else 8,
    "n_departamentos": int(df.groupby(["iso_a0", "adm_1_name"]).ngroups) if "adm_1_name" in df.columns else 169,
})
metrics_path = os.path.join(MODELS, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
s3.upload(metrics_path, s3.PREFIX_MODELOS + "metrics.json")
print("  [S3↑] metrics.json actualizado")

# ── Scatter data (muestra de 400 puntos representativos) ─────────────────────
iso_col  = df_test["iso_a0"].values  if "iso_a0"    in df_test.columns else ["?"] * len(y_real)
dept_col = df_test["adm_1_name"].values if "adm_1_name" in df_test.columns else ["?"] * len(y_real)
ano_col  = df_test["ano"].values
mes_col  = df_test["mes"].values

rng = np.random.default_rng(42)
n_sample = min(400, len(y_real))
idx_sample = rng.choice(len(y_real), size=n_sample, replace=False)

points_xgb = [
    {"actual": round(float(y_real[i]), 2), "pred": round(float(pred_xgb[i]), 2),
     "iso": str(iso_col[i]), "ano": int(ano_col[i])}
    for i in idx_sample
]
points_ens = [
    {"actual": round(float(y_real[i]), 2), "pred": round(float(pred_ens[i]), 2),
     "iso": str(iso_col[i]), "ano": int(ano_col[i])}
    for i in idx_sample
]
points_lstm = [
    {"actual": round(float(y_real[i]), 2), "pred": round(float(pred_lstm_arr[i] if valid[i] else pred_xgb[i]), 2),
     "iso": str(iso_col[i]), "ano": int(ano_col[i])}
    for i in idx_sample
]

scatter = {
    "ensemble": points_ens,
    "xgboost":  points_xgb,
    "lstm":     points_lstm,
    "metricas": {
        "ensemble": {"r2": round(r2_ens, 4),   "mae": round(mae_ens, 2),   "rmse": round(rmse_ens, 2)},
        "xgboost":  {"r2": round(r2_xgb_v, 4), "mae": round(mae_xgb_v, 2), "rmse": round(rmse_xgb_v, 2)},
        "lstm":     {"r2": round(r2_lstm_v, 4), "mae": round(mae_lstm_v, 2),"rmse": round(rmse_lstm_v, 2)},
    }
}

scatter_path = os.path.join(MODELS, "scatter_data.json")
with open(scatter_path, "w") as f:
    json.dump(scatter, f)
s3.upload(scatter_path, s3.PREFIX_MODELOS + "scatter_data.json")
print(f"  [S3↑] scatter_data.json ({n_sample} puntos)")
print("\nListo. Redespliega Railway para que el backend sirva los nuevos datos.")
