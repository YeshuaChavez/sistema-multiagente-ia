# -*- coding: utf-8 -*-
"""Calcula el R² real del ensemble alineando los test sets de LightGBM y LSTM."""
import os, sys, pickle, json
import numpy as np, pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final")
from Scripts.entrenar_y_guardar_modelos import (
    generar_lags_y_vecinos_dinamico, DengueLSTMModel
)

BASE_DIR  = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
DB_DIR    = os.path.join(BASE_DIR, "Base de Datos")
MODEL_DIR = os.path.join(DB_DIR, "modelos")

print("Cargando dataset y features...")
df_raw = pd.read_csv(os.path.join(DB_DIR, "datos_procesados", "dataset_maestro_mensual_latam.csv"))
df = generar_lags_y_vecinos_dinamico(df_raw, DB_DIR)
yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)

with open(os.path.join(MODEL_DIR, "cols_feat.pkl"), "rb") as f:
    COLS_FEAT = pickle.load(f)

df_test = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)].copy().reset_index(drop=True)

# ── LightGBM predictions ──────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    modelo_ml = pickle.load(f)
with open(os.path.join(MODEL_DIR, "imputador_ml.pkl"), "rb") as f:
    imputador_ml = pickle.load(f)
with open(os.path.join(MODEL_DIR, "escalador_ml.pkl"), "rb") as f:
    escalador_ml = pickle.load(f)

X_test_imp = pd.DataFrame(imputador_ml.transform(df_test[COLS_FEAT]), columns=COLS_FEAT)
X_test_esc = pd.DataFrame(escalador_ml.transform(X_test_imp), columns=COLS_FEAT)
pred_lgbm  = np.expm1(modelo_ml.predict(X_test_esc))

# Lookup (iso, dept, ano, mes) -> lgbm_pred, y_true
lgbm_lookup  = {}
y_true_lookup = {}
for i, row in df_test.iterrows():
    key = (row['iso_a0'], row['adm_1_name'], int(row['ano']), int(row['mes']))
    lgbm_lookup[key]   = float(pred_lgbm[i])
    y_true_lookup[key] = float(row['incidencia_dengue'])

print(f"  LightGBM test rows: {len(lgbm_lookup)}")

# ── LSTM sequences with identifiers ──────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "lstm_features.pkl"), "rb") as f:
    LSTM_FEATURES = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lstm_config.json"), "r") as f:
    lstm_config = json.load(f)
with open(os.path.join(MODEL_DIR, "escalador_lstm.pkl"), "rb") as f:
    escalador_lstm = pickle.load(f)

SEQ_LEN = lstm_config.get("seq_len", 12)

def create_sequences_with_ids(df, feat_cols, target_col, seq_len):
    X_list, y_list, ids = [], [], []
    for (iso, dept), grp in df.groupby(['iso_a0', 'adm_1_name']):
        grp = grp.sort_values(['ano', 'mes']).reset_index(drop=True)
        if len(grp) < seq_len + 1:
            continue
        for i in range(seq_len, len(grp)):
            X_list.append(grp[feat_cols].iloc[i - seq_len:i].values)
            y_list.append(grp[target_col].iloc[i])
            ids.append((iso, dept, int(grp['ano'].iloc[i]), int(grp['mes'].iloc[i])))
    return np.array(X_list), np.array(y_list), ids

X_all, y_all, ids_all = create_sequences_with_ids(df, LSTM_FEATURES, 'incidencia_dengue', SEQ_LEN)
anos_arr  = np.array([x[2] for x in ids_all])
test_mask = anos_arr >= 2021

X_test_seq = X_all[test_mask]
y_test_seq = y_all[test_mask]
ids_test   = [ids_all[i] for i in range(len(ids_all)) if test_mask[i]]

X_test_sc = escalador_lstm.transform(
    X_test_seq.reshape(-1, len(LSTM_FEATURES))
).reshape(X_test_seq.shape)

model_lstm = DengueLSTMModel(input_dim=len(LSTM_FEATURES))
model_lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"), map_location='cpu'))
model_lstm.eval()
with torch.no_grad():
    pred_lstm = np.expm1(model_lstm(torch.tensor(X_test_sc, dtype=torch.float32)).numpy().flatten())

print(f"  LSTM test sequences: {len(ids_test)}")

# ── Align by (iso, dept, ano, mes) ───────────────────────────────────────────
y_aln, lgbm_aln, lstm_aln = [], [], []
for i, key in enumerate(ids_test):
    if key in lgbm_lookup:
        y_aln.append(y_true_lookup[key])
        lgbm_aln.append(lgbm_lookup[key])
        lstm_aln.append(float(pred_lstm[i]))

y_aln    = np.array(y_aln)
lgbm_aln = np.array(lgbm_aln)
lstm_aln = np.array(lstm_aln)
ens_aln  = (lgbm_aln + lstm_aln) / 2

print(f"  Rows alineados:  {len(y_aln)}")
print()
print("=== MÉTRICAS FINALES (test 2021-2022) ===")
print(f"  LightGBM (alineado)  R²={r2_score(y_aln, lgbm_aln):.4f}  MAE={mean_absolute_error(y_aln, lgbm_aln):.4f}")
print(f"  LSTM     (alineado)  R²={r2_score(y_aln, lstm_aln):.4f}  MAE={mean_absolute_error(y_aln, lstm_aln):.4f}")
print(f"  Ensemble (2-way)     R²={r2_score(y_aln, ens_aln):.4f}  MAE={mean_absolute_error(y_aln, ens_aln):.4f}")

# Actualizar metrics.json con valores reales
r2_ens  = round(float(r2_score(y_aln, ens_aln)), 4)
mae_ens = round(float(mean_absolute_error(y_aln, ens_aln)), 4)
r2_lgbm_full = round(float(r2_score(y_aln, lgbm_aln)), 4)
mae_lgbm_full = round(float(mean_absolute_error(y_aln, lgbm_aln)), 4)
r2_lstm_full  = round(float(r2_score(y_aln, lstm_aln)), 4)
mae_lstm_full = round(float(mean_absolute_error(y_aln, lstm_aln)), 4)

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)

metrics.update({
    "r2_lgbm": r2_lgbm_full,
    "mae_lgbm": mae_lgbm_full,
    "r2_lstm": r2_lstm_full,
    "mae_lstm": mae_lstm_full,
    "r2_ensemble": r2_ens,
    "mae_ensemble": mae_ens,
})
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"\n  metrics.json actualizado: ensemble R²={r2_ens}, MAE={mae_ens}")
