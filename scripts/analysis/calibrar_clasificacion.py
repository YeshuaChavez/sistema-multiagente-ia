# -*- coding: utf-8 -*-
"""
Calibra los umbrales de clasificación del ensemble sobre el validation set (2020)
para corregir el sesgo sistemático del modelo. Los umbrales se buscan sobre las
predicciones (no sobre los datos reales) minimizando el error de clasificación.

Resultado: thresholds_clasificacion.json → usado por el backend para clasificar.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from scipy.optimize import minimize

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODELS = os.path.join(BASE, 'Base de Datos', 'modelos')
PROC   = os.path.join(BASE, 'Base de Datos', 'datos_procesados')

df       = pd.read_csv(os.path.join(PROC, 'dataset_features_latam.csv'))
df_master= pd.read_csv(os.path.join(PROC, 'dataset_maestro_mensual_latam.csv'))

with open(os.path.join(MODELS, 'cols_feat.pkl'), 'rb') as f:   cols_feat   = pickle.load(f)
with open(os.path.join(MODELS, 'metrics.json'))         as f:   metrics     = json.load(f)
with open(os.path.join(MODELS, 'pipeline_ml.pkl'), 'rb') as f:  pipe_xgb    = pickle.load(f)
with open(os.path.join(MODELS, 'lstm_config.json'))     as f:   lstm_cfg    = json.load(f)
with open(os.path.join(MODELS, 'lstm_features.pkl'), 'rb') as f: lstm_feats  = pickle.load(f)
with open(os.path.join(MODELS, 'escalador_lstm.pkl'), 'rb') as f: scaler_lstm = pickle.load(f)

w_xgb  = metrics.get('ensemble_w_xgb',  0.90)
w_lstm = metrics.get('ensemble_w_lstm', 0.10)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

_sd = torch.load(os.path.join(MODELS, 'lstm_model.pth'), map_location='cpu', weights_only=True)
_nl = sum(1 for k in _sd if k.startswith('lstm.weight_ih_l'))
model_lstm = LSTMModel(lstm_cfg['input_dim'], lstm_cfg['hidden_dim'], _nl, lstm_cfg.get('dropout', 0.2))
model_lstm.load_state_dict(_sd)
model_lstm.eval()
SEQ_LEN = lstm_cfg.get('seq_len', 12)

# ── Percentiles locales por departamento (train 2014-2020) ────────────────────
df_hist = df_master[df_master['ano'] <= 2020].copy()
df_hist['iso_u'] = df_hist['iso_a0'].str.strip().str.upper()
df_hist['adm_u'] = df_hist['adm_1_name'].str.strip().str.upper()
g25 = float(df_hist['incidencia_dengue'].quantile(0.25))
g50 = float(df_hist['incidencia_dengue'].quantile(0.50))
g90 = float(df_hist['incidencia_dengue'].quantile(0.90))

pct_map = {}
for (iso, adm), grp in df_hist.groupby(['iso_u', 'adm_u']):
    inc = grp['incidencia_dengue']
    pct_map[(iso, adm)] = (
        float(inc.quantile(0.25)),
        max(float(inc.quantile(0.50)), 0.5),
        max(float(inc.quantile(0.90)), g90),
    )

def clase_real(valor, iso, adm):
    p25, p50, p90 = pct_map.get((iso, adm), (g25, g50, g90))
    if valor <= p25:   return 0
    elif valor <= p50: return 1
    elif valor <= p90: return 2
    else:              return 3

def get_predictions(anos):
    df_split  = df[df['ano'].isin(anos)].copy()
    pred_xgb  = np.expm1(pipe_xgb.predict(df_split[cols_feat]))
    df_scaled = df.copy()
    df_scaled[lstm_feats] = scaler_lstm.transform(df[lstm_feats])
    pred_lstm = []
    for idx in df_split.index:
        pos = df.index.get_loc(idx)
        if pos < SEQ_LEN:
            pred_lstm.append(np.nan); continue
        seq = df_scaled.iloc[pos - SEQ_LEN:pos][lstm_feats].values
        with torch.no_grad():
            p = model_lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)).item()
        pred_lstm.append(max(0.0, np.expm1(p)))
    pred_lstm = np.array(pred_lstm, dtype=float)
    valid     = ~np.isnan(pred_lstm)
    pred_ens  = w_xgb * pred_xgb + w_lstm * np.where(valid, pred_lstm, pred_xgb)

    y_real, y_clase = [], []
    for row, pe in zip(df_split.itertuples(), pred_ens):
        iso = str(row.iso_a0).strip().upper()
        adm = str(row.adm_1_name).strip().upper()
        y_real.append(row.incidencia_dengue)
        y_clase.append(clase_real(row.incidencia_dengue, iso, adm))

    return np.array(pred_ens), np.array(y_real), np.array(y_clase)

# ── Validation set: 2020 ──────────────────────────────────────────────────────
print("[1/3] Calculando predicciones en val 2020 y test 2021-2022...")
pred_val, y_val, y_clase_val = get_predictions([2020])
pred_test, y_test, y_clase_test = get_predictions([2021, 2022])

# ── Umbrales actuales (percentiles de datos reales) ───────────────────────────
def clasificar_con_umbrales(preds, t1, t2, t3):
    clases = np.zeros(len(preds), dtype=int)
    clases[preds > t1] = 1
    clases[preds > t2] = 2
    clases[preds > t3] = 3
    return clases

t1_actual = np.percentile(y_val, 25)
t2_actual = np.percentile(y_val, 50)
t3_actual = np.percentile(y_val, 90)
acc_actual_val  = accuracy_score(y_clase_val,  clasificar_con_umbrales(pred_val,  t1_actual, t2_actual, t3_actual))
acc_actual_test = accuracy_score(y_clase_test, clasificar_con_umbrales(pred_test, t1_actual, t2_actual, t3_actual))
print(f"  Umbrales actuales  (p25/p50/p90 de y_real): acc_val={acc_actual_val*100:.1f}%  acc_test={acc_actual_test*100:.1f}%")

# ── Optimizar umbrales sobre predicciones del val set ────────────────────────
print("[2/3] Optimizando umbrales sobre predicciones del val 2020...")

def neg_kappa(params):
    t1, t2, t3 = sorted(params)  # forzar orden
    pred_clase = clasificar_con_umbrales(pred_val, t1, t2, t3)
    return -cohen_kappa_score(y_clase_val, pred_clase)

# Punto de partida: percentiles de las predicciones del val set
x0 = [np.percentile(pred_val, 25),
      np.percentile(pred_val, 50),
      np.percentile(pred_val, 90)]

res = minimize(neg_kappa, x0, method='Nelder-Mead',
               options={'maxiter': 5000, 'xatol': 0.01, 'fatol': 1e-5})

t1_opt, t2_opt, t3_opt = sorted(res.x)
acc_opt_val  = accuracy_score(y_clase_val,  clasificar_con_umbrales(pred_val,  t1_opt, t2_opt, t3_opt))
kappa_opt_val= cohen_kappa_score(y_clase_val, clasificar_con_umbrales(pred_val, t1_opt, t2_opt, t3_opt))
print(f"  Umbrales optimizados:                        acc_val={acc_opt_val*100:.1f}%  kappa={kappa_opt_val:.3f}")
print(f"  Umbrales: T1={t1_opt:.2f}  T2={t2_opt:.2f}  T3={t3_opt:.2f}")

# ── Evaluar en test set ───────────────────────────────────────────────────────
print("[3/3] Evaluando en test 2021-2022 con umbrales calibrados...")
CLASES = ['Normal', 'Vigilancia', 'Alerta', 'Epidemia']
pred_clase_test = clasificar_con_umbrales(pred_test, t1_opt, t2_opt, t3_opt)
acc_opt_test   = accuracy_score(y_clase_test, pred_clase_test)
kappa_opt_test = cohen_kappa_score(y_clase_test, pred_clase_test)

epi_real = (y_clase_test == 3)
epi_pred = (pred_clase_test == 3)
tp = (epi_real & epi_pred).sum()
fn = (epi_real & ~epi_pred).sum()
fp = (~epi_real & epi_pred).sum()
recall_epi    = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_epi = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_epi        = 2*precision_epi*recall_epi / (precision_epi+recall_epi) if (precision_epi+recall_epi) > 0 else 0

print(f"\n{'='*55}")
print(f"  CLASIFICACIÓN CALIBRADA — Test 2021-2022")
print(f"{'='*55}")
print(f"  Accuracy : {acc_opt_test*100:.1f}%  (antes: 57.2%)")
print(f"  Kappa    : {kappa_opt_test:.3f}  (antes: 0.411)")
print(f"\n{classification_report(y_clase_test, pred_clase_test, target_names=CLASES, digits=3)}")
print(f"  EPIDEMIA:")
print(f"    Recall    : {recall_epi*100:.1f}%  (antes: 58.0%)")
print(f"    Precision : {precision_epi*100:.1f}%  (antes: 86.5%)")
print(f"    F1        : {f1_epi*100:.1f}%  (antes: 69.5%)")
print(f"    Detectados: {tp} de {epi_real.sum()} | Falsos: {fp}")

# ── Guardar umbrales ──────────────────────────────────────────────────────────
thresholds = {
    "t1_normal_vigilancia": round(float(t1_opt), 4),
    "t2_vigilancia_alerta": round(float(t2_opt), 4),
    "t3_alerta_epidemia":   round(float(t3_opt), 4),
    "acc_val":   round(float(acc_opt_val),  4),
    "kappa_val": round(float(kappa_opt_val), 4),
    "acc_test":  round(float(acc_opt_test),  4),
    "kappa_test":round(float(kappa_opt_test),4),
}
out_path = os.path.join(MODELS, 'thresholds_clasificacion.json')
with open(out_path, 'w') as f:
    json.dump(thresholds, f, indent=4)
s3.upload(out_path, s3.PREFIX_MODELOS + 'thresholds_clasificacion.json')
print(f"\n  thresholds_clasificacion.json guardado y subido a S3.")
