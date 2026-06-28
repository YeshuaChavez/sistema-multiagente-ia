"""
Compara estrategias de pesos para el ensemble XGBoost + LSTM.
No entrena nada — solo inference + optimización de pesos.
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize_scalar, minimize
import boto3

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data", "processed", "dataset_features_latam.csv")
TMP    = os.path.join(BASE, "data", "models")

AWS_KEY    = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = "us-east-2"
BUCKET     = "epipredict-dengue"

SEQ_LEN    = 12
SEMILLA    = 42
LSTM_FEATS = ["tmax_promedio","tmin_promedio","precipitacion",
              "humedad_promedio","agua_basica","incidencia_dengue"]
COLS_EXCLUIR = ["iso_a0","pais","adm_1_name","ano","mes",
                "casos_dengue","poblacion","incidencia_dengue"]

# ── Modelo LSTM ───────────────────────────────────────────────────────────────
class DengueLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def build_sequences(df, features, seq_len):
    X_list, y_list, ids = [], [], []
    for (iso, adm), grp in df.groupby(["iso_a0", "adm_1_name"]):
        grp = grp.sort_values(["ano","mes"]).reset_index(drop=True)
        vals = grp[features].values
        inc  = grp["incidencia_dengue"].values
        anos = grp["ano"].values
        meses= grp["mes"].values
        for i in range(seq_len, len(grp)):
            X_list.append(vals[i-seq_len:i])
            y_list.append(inc[i])
            ids.append((str(iso).upper(), str(adm).upper(), int(anos[i]), int(meses[i])))
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            ids)

# ── Descargar modelos de S3 si no existen ─────────────────────────────────────
def descargar_si_falta(s3, fname):
    local = os.path.join(TMP, fname)
    if not os.path.exists(local):
        print(f"  Descargando {fname}...")
        s3.download_file(BUCKET, f"modelos/{fname}", local)
    return local

print("=== Comparación de estrategias de pesos Ensemble ===\n")
s3 = boto3.client("s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=AWS_REGION)

os.makedirs(TMP, exist_ok=True)
path_pipe  = descargar_si_falta(s3, "pipeline_ml.pkl")
path_lstm  = descargar_si_falta(s3, "lstm_model.pth")
path_esc   = descargar_si_falta(s3, "escalador_lstm.pkl")
path_cfg   = descargar_si_falta(s3, "lstm_config.json")

# ── Cargar dataset ─────────────────────────────────────────────────────────────
print("Cargando dataset...")
df = pd.read_csv(DATA)
COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
print(f"  {len(df)} registros | {len(COLS_FEAT)} features")

df_test = df[df["ano"] >= 2021].copy()
df_val  = df[df["ano"].isin([2018, 2019, 2020])].copy()

# ── Cargar XGBoost ─────────────────────────────────────────────────────────────
print("Cargando XGBoost pipeline...")
with open(path_pipe, "rb") as f:
    pipe_xgb = pickle.load(f)

xgb_pred_test = pipe_xgb.predict(df_test[COLS_FEAT])   # log1p scale
xgb_pred_val  = pipe_xgb.predict(df_val[COLS_FEAT])

# ── Cargar LSTM ────────────────────────────────────────────────────────────────
print("Cargando LSTM...")
with open(path_cfg) as f: cfg = json.load(f)
with open(path_esc, "rb") as f: escalador = pickle.load(f)

device = torch.device("cpu")
model = DengueLSTMModel(6, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
model.load_state_dict(torch.load(path_lstm, map_location=device))
model.eval()

# Secuencias test (>=2021)
X_seq_all, y_seq_all, ids_all = build_sequences(df, LSTM_FEATS, SEQ_LEN)
X_seq_all_sc = escalador.transform(
    X_seq_all.reshape(-1,6)).reshape(X_seq_all.shape)
anos_all = np.array([i[2] for i in ids_all])

mask_test = anos_all >= 2021
mask_val  = np.array([i[2] for i in ids_all]) >= 2018
mask_val &= anos_all <= 2020

with torch.no_grad():
    lstm_pred_test = model(torch.tensor(X_seq_all_sc[mask_test], dtype=torch.float32)).numpy()
    lstm_pred_val  = model(torch.tensor(X_seq_all_sc[mask_val],  dtype=torch.float32)).numpy()

# ── Alinear test XGB ↔ LSTM por (iso, adm, ano, mes) ─────────────────────────
print("Alineando predicciones XGB <-> LSTM por (iso, adm, ano, mes)...")
ids_test = [i for i in ids_all if i[2] >= 2021]
xgb_lkp  = {
    (str(r.iso_a0).upper(), str(r.adm_1_name).upper(), int(r.ano), int(r.mes)): float(p)
    for r, p in zip(df_test.itertuples(), xgb_pred_test)
}

y_te, xgb_te, lstm_te = [], [], []
for sid, lp in zip(ids_test, lstm_pred_test):
    xp = xgb_lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[
        (df_test["iso_a0"].str.upper()==iso) &
        (df_test["adm_1_name"].str.upper()==adm) &
        (df_test["ano"]==ano) & (df_test["mes"]==mes)]
    if len(row):
        y_te.append(float(row["incidencia_dengue"].iloc[0]))
        xgb_te.append(xp); lstm_te.append(float(lp))

y_te    = np.array(y_te)
ly_te   = np.log1p(y_te)
xgb_te  = np.array(xgb_te)
lstm_te = np.array(lstm_te)

# Alinear val
ids_val = [i for i in ids_all if 2018 <= i[2] <= 2020]
xgb_lkp_v = {
    (str(r.iso_a0).upper(), str(r.adm_1_name).upper(), int(r.ano), int(r.mes)): float(p)
    for r, p in zip(df_val.itertuples(), xgb_pred_val)
}
y_v, xgb_v, lstm_v = [], [], []
for sid, lp in zip(ids_val, lstm_pred_val):
    xp = xgb_lkp_v.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_val[
        (df_val["iso_a0"].str.upper()==iso) &
        (df_val["adm_1_name"].str.upper()==adm) &
        (df_val["ano"]==ano) & (df_val["mes"]==mes)]
    if len(row):
        y_v.append(float(row["incidencia_dengue"].iloc[0]))
        xgb_v.append(xp); lstm_v.append(float(lp))

y_v    = np.array(y_v)
ly_v   = np.log1p(y_v)
xgb_v  = np.array(xgb_v)
lstm_v = np.array(lstm_v)
print(f"  Test alineado: {len(y_te)} | Val alineado: {len(y_v)}\n")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Métricas individuales ─────────────────────────────────────────────────────
r2_xgb  = r2_score(ly_te, xgb_te)
mae_xgb = mean_absolute_error(y_te, np.expm1(xgb_te))
rmse_xgb= np.sqrt(mean_squared_error(y_te, np.expm1(xgb_te)))

r2_lstm  = r2_score(ly_te, lstm_te)
mae_lstm = mean_absolute_error(y_te, np.expm1(lstm_te))
rmse_lstm= np.sqrt(mean_squared_error(y_te, np.expm1(lstm_te)))

def ens_metrics(w):
    pred_log  = w*xgb_te + (1-w)*lstm_te
    pred_real = np.maximum(np.expm1(pred_log), 0)
    return (r2_score(ly_te, pred_log),
            mean_absolute_error(y_te, pred_real),
            np.sqrt(mean_squared_error(y_te, pred_real)))

# ── Estrategias de peso ────────────────────────────────────────────────────────
def opt_r2(bounds=(0.0,1.0)):
    res = minimize_scalar(lambda w: -r2_score(ly_v, w*xgb_v+(1-w)*lstm_v),
                          bounds=bounds, method="bounded")
    return float(res.x)

def opt_mae(bounds=(0.0,1.0)):
    res = minimize_scalar(
        lambda w: mean_absolute_error(y_v, np.maximum(np.expm1(w*xgb_v+(1-w)*lstm_v),0)),
        bounds=bounds, method="bounded")
    return float(res.x)

def opt_rmse(bounds=(0.0,1.0)):
    res = minimize_scalar(
        lambda w: np.sqrt(mean_squared_error(y_v, np.maximum(np.expm1(w*xgb_v+(1-w)*lstm_v),0))),
        bounds=bounds, method="bounded")
    return float(res.x)

w_r2_free   = opt_r2(bounds=(0.0, 1.0))
w_mae_free  = opt_mae(bounds=(0.0, 1.0))
w_rmse_free = opt_rmse(bounds=(0.0, 1.0))
w_prop      = r2_xgb / (r2_xgb + r2_lstm)  # proporcional al R² en test
w_equal     = 0.5

estrategias = [
    ("Individal XGBoost",          1.0),
    ("Individual LSTM",            0.0),
    ("──────────────────",         None),
    ("Optim. R²  (libre 0-1)",     w_r2_free),
    ("Optim. MAE (libre 0-1)",     w_mae_free),
    ("Optim. RMSE(libre 0-1)",     w_rmse_free),
    ("Proporcional al R²",         w_prop),
    ("Igual 50/50",                w_equal),
    ("Original (R² val, 0.4-0.95)",0.95),
]

print(f"{'Estrategia':<32} {'w_xgb':>6}  {'R²':>7}  {'MAE':>6}  {'RMSE':>7}")
print("─"*70)
for nombre, w in estrategias:
    if w is None:
        print(f"  {nombre}")
        continue
    r2, mae, rmse = ens_metrics(w)
    marker = ""
    print(f"  {nombre:<30} {w:>6.3f}  {r2*100:>6.2f}%  {mae:>6.2f}  {rmse:>7.2f}  {marker}")

print("\n=== MEJOR POR MÉTRICA ===")
resultados = {n: (w, *ens_metrics(w)) for n,w in estrategias if w is not None}
mejor_r2   = max(resultados.items(), key=lambda x: x[1][1])
mejor_mae  = min(resultados.items(), key=lambda x: x[1][2])
mejor_rmse = min(resultados.items(), key=lambda x: x[1][3])
print(f"  Mejor R²  : {mejor_r2[0]}  w={mejor_r2[1][0]:.3f}  R²={mejor_r2[1][1]*100:.2f}%")
print(f"  Mejor MAE : {mejor_mae[0]}  w={mejor_mae[1][0]:.3f}  MAE={mejor_mae[1][2]:.2f}")
print(f"  Mejor RMSE: {mejor_rmse[0]}  w={mejor_rmse[1][0]:.3f}  RMSE={mejor_rmse[1][3]:.2f}")

# ── Clasificación 3 etiquetas ─────────────────────────────────────────────────
print("\n\n=== CLASIFICACIÓN 3 ETIQUETAS (Endemico / Alerta / Epidemia) ===")
print("    Umbrales por departamento: p50 y p90 del historial de entrenamiento (<=2020)\n")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

df_hist = df[df["ano"] <= 2020].copy()

# Percentiles por (iso, adm)
dept_percentiles = {}
for (iso, adm), grp in df_hist.groupby(["iso_a0", "adm_1_name"]):
    key = (str(iso).upper(), str(adm).upper())
    inc = grp["incidencia_dengue"].values
    dept_percentiles[key] = (
        max(float(np.percentile(inc, 50)), 0.5),
        max(float(np.percentile(inc, 90)), 5.0),
    )

# Percentiles globales como fallback
p50_global = max(float(df_hist["incidencia_dengue"].quantile(0.50)), 0.5)
p90_global = max(float(df_hist["incidencia_dengue"].quantile(0.90)), 5.0)

def clasificar(val, iso, adm):
    p50, p90 = dept_percentiles.get((iso, adm), (p50_global, p90_global))
    if val <= p50:   return 0  # Endemico
    elif val <= p90: return 1  # Alerta
    else:            return 2  # Epidemia

# Etiquetas verdaderas para test set alineado
ids_te_list = [i for i in ids_all if i[2] >= 2021]
# Solo los alineados (mismos que y_te)
# Reconstruir ids alineados en el mismo orden
ids_alineados = []
for sid, lp in zip([i for i in ids_all if i[2] >= 2021], lstm_pred_test):
    xp = xgb_lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[
        (df_test["iso_a0"].str.upper()==iso) &
        (df_test["adm_1_name"].str.upper()==adm) &
        (df_test["ano"]==ano) & (df_test["mes"]==mes)]
    if len(row):
        ids_alineados.append((iso, adm))

y_true_cls = np.array([clasificar(v, iso, adm)
                       for v,(iso,adm) in zip(y_te, ids_alineados)])

etiquetas  = ["Endemico", "Alerta", "Epidemia"]
dist = np.bincount(y_true_cls, minlength=3)
print(f"  Distribucion real test: Endemico={dist[0]} | Alerta={dist[1]} | Epidemia={dist[2]}\n")

def cls_metrics(w, label):
    pred_log  = w*xgb_te + (1-w)*lstm_te
    pred_real = np.maximum(np.expm1(pred_log), 0)
    y_pred_cls = np.array([clasificar(v, iso, adm)
                            for v,(iso,adm) in zip(pred_real, ids_alineados)])
    acc   = accuracy_score(y_true_cls, y_pred_cls)
    kappa = cohen_kappa_score(y_true_cls, y_pred_cls)
    report = classification_report(y_true_cls, y_pred_cls,
                                   target_names=etiquetas, output_dict=True, zero_division=0)
    print(f"  --- {label} (w_xgb={w:.2f}) ---")
    print(f"  Accuracy={acc*100:.1f}%   Kappa={kappa:.3f}")
    print(f"  {'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Soporte':>9}")
    for cls in etiquetas:
        r = report[cls]
        print(f"  {cls:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f} {int(r['support']):>9}")
    print()

cls_metrics(0.95, "Original 95/5")
cls_metrics(0.50, "50/50 Proporcional")
cls_metrics(1.00, "Solo XGBoost")
cls_metrics(0.00, "Solo LSTM")
