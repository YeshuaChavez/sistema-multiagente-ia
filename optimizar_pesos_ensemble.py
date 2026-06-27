# -*- coding: utf-8 -*-
"""
Optimiza los pesos del ensemble en un validation set (año 2020)
usando scipy.optimize. Los pesos se buscan en log1p-space para consistencia
con el entrenamiento. El test set (2021-2022) nunca se toca durante la búsqueda.

Flujo:
  1. Val 2020  → minimize_scalar(neg R²) → w_xgb_opt
  2. Test 2021-2022 → evaluar con w_xgb_opt → reportar métricas finales
  3. Actualizar metrics.json en disco y S3.

Siguiente paso: correr generar_scatter_data.py para reflejar los nuevos pesos
en scatter_data.json (usado por el frontend).
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize_scalar
sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3
from agente_4_prediccion_dl import DengueLSTMModel, _build_sequences, LSTM_SEQ_LEN
import pandas as pd

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODEL_DIR = os.path.join(BASE, 'Base de Datos', 'modelos')
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

# ── Cargar dataset ────────────────────────────────────────────────────────────
print("[1/5] Cargando dataset...")
df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT    = [c for c in df.columns if c not in COLS_EXCLUIR]

# ── Cargar XGBoost ────────────────────────────────────────────────────────────
print("[2/5] Cargando modelos...")
with open(os.path.join(MODEL_DIR, 'pipeline_ml.pkl'), 'rb') as f:
    pipe_xgb = pickle.load(f)

# ── Cargar LSTM ───────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'lstm_config.json')) as f:
    cfg = json.load(f)
with open(os.path.join(MODEL_DIR, 'escalador_lstm.pkl'), 'rb') as f:
    sc_lstm = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'lstm_features.pkl'), 'rb') as f:
    lstm_feats = pickle.load(f)

lstm_model = DengueLSTMModel(cfg['input_dim'], cfg['hidden_dim'], dropout=cfg['dropout'])
lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'lstm_model.pth'), map_location='cpu'))
lstm_model.eval()

# ── Secuencias LSTM para todo el dataset (una sola pasada) ────────────────────
print("[3/5] Construyendo secuencias LSTM...")
X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, lstm_feats, LSTM_SEQ_LEN)


def _get_aligned_preds(year_mask):
    """Devuelve (y, pred_xgb_log, pred_lstm_log) alineados por clave (iso,adm,ano,mes)."""
    df_split = df[year_mask].copy()
    pxgb_log = pipe_xgb.predict(df_split[COLS_FEAT])

    anos_target = set(int(a) for a in df_split['ano'].unique())
    lstm_mask = np.array([sid[2] in anos_target for sid in seq_ids])
    X_s   = X_seq[lstm_mask]
    ids_s = [seq_ids[i] for i, m in enumerate(lstm_mask) if m]
    X_sc  = sc_lstm.transform(X_s.reshape(-1, len(lstm_feats))).reshape(X_s.shape)
    with torch.no_grad():
        plstm_log = lstm_model(torch.tensor(X_sc, dtype=torch.float32)).numpy().flatten()

    xgb_lkp = {
        (str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
        for r, p in zip(df_split.itertuples(), pxgb_log)
    }
    y_l, xgb_l, lstm_l = [], [], []
    for sid, lp in zip(ids_s, plstm_log):
        xp = xgb_lkp.get(sid)
        if xp is None:
            continue
        iso, adm, ano, mes = sid
        row = df_split[
            (df_split['iso_a0'].str.upper() == iso) &
            (df_split['adm_1_name'].str.upper() == adm) &
            (df_split['ano'] == ano) &
            (df_split['mes'] == mes)
        ]
        if len(row):
            y_l.append(float(row['incidencia_dengue'].iloc[0]))
            xgb_l.append(xp)
            lstm_l.append(lp)

    return np.array(y_l), np.array(xgb_l), np.array(lstm_l)


# ── Validation set: año 2020 ──────────────────────────────────────────────────
print("[4/5] Optimizando pesos en validation set (2020)...")
y_v, lxgb_v, llstm_v = _get_aligned_preds(df['ano'] == 2020)
ly_v = np.log1p(y_v)
# lxgb_v ya está en log1p (salida directa del pipeline)
# llstm_v ya está en log1p (salida directa del modelo LSTM)

def neg_r2_val(w):
    ens = w * lxgb_v + (1.0 - w) * llstm_v
    return -r2_score(ly_v, ens)

res = minimize_scalar(neg_r2_val, bounds=(0.40, 0.90), method='bounded')
w_xgb_opt  = float(res.x)
w_lstm_opt = 1.0 - w_xgb_opt
r2_val_old = -neg_r2_val(0.5117)  # referencia con pesos anteriores
r2_val_new = -res.fun

print(f"  Pesos anteriores (proporcional R²): w_xgb=0.5117 → R² val={r2_val_old*100:.2f}%")
print(f"  Pesos optimizados:                  w_xgb={w_xgb_opt:.4f} → R² val={r2_val_new*100:.2f}%")

# ── Test set: 2021-2022 con pesos optimizados ─────────────────────────────────
print("[5/5] Evaluando en test set (2021-2022)...")
y_t, lxgb_t, llstm_t = _get_aligned_preds(df['ano'] >= 2021)
ly_t = np.log1p(y_t)

r2_xgb  = r2_score(ly_t, lxgb_t)
mae_xgb = mean_absolute_error(y_t, np.expm1(lxgb_t))
r2_lstm  = r2_score(ly_t, llstm_t)
mae_lstm = mean_absolute_error(y_t, np.expm1(llstm_t))

ens_log = w_xgb_opt * lxgb_t + w_lstm_opt * llstm_t
ens_raw = np.expm1(ens_log)
r2_ens  = r2_score(ly_t, ens_log)
mae_ens = mean_absolute_error(y_t, ens_raw)

print(f"\n{'='*58}")
print(f"  ENSEMBLE OPTIMIZADO — pesos hallados en val 2020")
print(f"  w_xgb={w_xgb_opt:.4f}  w_lstm={w_lstm_opt:.4f}")
print(f"  XGBoost  : R²={r2_xgb*100:.2f}%  MAE={mae_xgb:.2f}")
print(f"  LSTM     : R²={r2_lstm*100:.2f}%  MAE={mae_lstm:.2f}")
print(f"  Ensemble : R²={r2_ens*100:.2f}%  MAE={mae_ens:.2f}  (antes: 89.79%)")
print(f"{'='*58}")

# ── Actualizar metrics.json ───────────────────────────────────────────────────
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
with open(metrics_path) as f:
    metrics = json.load(f)

metrics.update({
    'r2_xgb':          round(r2_xgb, 4),
    'mae_xgb':         round(mae_xgb, 4),
    'r2_lstm':         round(r2_lstm, 4),
    'mae_lstm':        round(mae_lstm, 4),
    'r2_ensemble':     round(r2_ens, 4),
    'mae_ensemble':    round(mae_ens, 4),
    'ensemble_w_xgb':  round(w_xgb_opt, 4),
    'ensemble_w_lstm': round(w_lstm_opt, 4),
})

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

s3.upload(metrics_path, s3.PREFIX_MODELOS + 'metrics.json')
print(f"\nmetrics.json actualizado y subido a S3.")
print("Siguiente paso: corre generar_scatter_data.py para actualizar scatter_data.json.")
