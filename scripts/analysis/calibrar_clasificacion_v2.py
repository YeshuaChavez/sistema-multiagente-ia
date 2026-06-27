# -*- coding: utf-8 -*-
"""
Calibración de umbrales por objetivo independiente:
  T1: maximizar recall de Normal  (el mayor problema actual)
  T2: sin cambio
  T3: maximizar F1 Epidemia con restricción precision >= 0.80
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODELS = os.path.join(BASE, 'Base de Datos', 'modelos')
PROC   = os.path.join(BASE, 'Base de Datos', 'datos_procesados')

df       = pd.read_csv(os.path.join(PROC, 'dataset_features_latam.csv'))
df_master= pd.read_csv(os.path.join(PROC, 'dataset_maestro_mensual_latam.csv'))

with open(os.path.join(MODELS, 'cols_feat.pkl'),    'rb') as f: cols_feat   = pickle.load(f)
with open(os.path.join(MODELS, 'metrics.json'))         as f: metrics     = json.load(f)
with open(os.path.join(MODELS, 'pipeline_ml.pkl'),  'rb') as f: pipe_xgb    = pickle.load(f)
with open(os.path.join(MODELS, 'lstm_config.json'))     as f: lstm_cfg    = json.load(f)
with open(os.path.join(MODELS, 'lstm_features.pkl'),'rb') as f: lstm_feats  = pickle.load(f)
with open(os.path.join(MODELS, 'escalador_lstm.pkl'),'rb') as f: scaler_lstm = pickle.load(f)

w_xgb = metrics.get('ensemble_w_xgb', 0.90)
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

# ── Percentiles locales (train 2014-2020) ─────────────────────────────────────
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

def clasificar(preds, t1, t2, t3):
    c = np.zeros(len(preds), dtype=int)
    c[preds > t1] = 1
    c[preds > t2] = 2
    c[preds > t3] = 3
    return c

def get_predictions(anos):
    df_s = df[df['ano'].isin(anos)].copy()
    px   = np.expm1(pipe_xgb.predict(df_s[cols_feat]))
    df_sc = df.copy()
    df_sc[lstm_feats] = scaler_lstm.transform(df[lstm_feats])
    pl = []
    for idx in df_s.index:
        pos = df.index.get_loc(idx)
        if pos < SEQ_LEN: pl.append(np.nan); continue
        seq = df_sc.iloc[pos-SEQ_LEN:pos][lstm_feats].values
        with torch.no_grad():
            p = model_lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)).item()
        pl.append(max(0.0, np.expm1(p)))
    pl   = np.array(pl, dtype=float)
    valid = ~np.isnan(pl)
    pe   = w_xgb * px + w_lstm * np.where(valid, pl, px)
    yc   = np.array([clase_real(r.incidencia_dengue,
                                str(r.iso_a0).strip().upper(),
                                str(r.adm_1_name).strip().upper())
                     for r in df_s.itertuples()])
    return pe, yc

print("[1/3] Calculando predicciones...")
pred_val,  yc_val  = get_predictions([2020])
pred_test, yc_test = get_predictions([2021, 2022])

# Umbrales base: percentiles de las predicciones del val set
t2_base = float(np.percentile(pred_val, 50))

# ── Optimizar T1 — maximizar recall de Normal ─────────────────────────────────
print("[2/3] Optimizando T1 (Normal) y T3 (Epidemia)...")
best_t1, best_recall_n = 0, -1
for t1 in np.linspace(np.percentile(pred_val, 10), np.percentile(pred_val, 60), 200):
    pc = clasificar(pred_val, t1, t2_base, 999)
    recall_n = ((yc_val == 0) & (pc == 0)).sum() / (yc_val == 0).sum()
    if recall_n > best_recall_n:
        best_recall_n = recall_n
        best_t1 = t1

# ── Optimizar T3 — maximizar F1 Epidemia con precision >= 0.80 ────────────────
best_t3, best_f1_epi = 999, -1
for t3 in np.linspace(np.percentile(pred_val, 70), np.percentile(pred_val, 99), 300):
    pc = clasificar(pred_val, best_t1, t2_base, t3)
    epi_r = (yc_val == 3); epi_p = (pc == 3)
    tp = (epi_r & epi_p).sum(); fp = (~epi_r & epi_p).sum(); fn = (epi_r & ~epi_p).sum()
    if (tp + fp) == 0: continue
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    if prec >= 0.80 and f1 > best_f1_epi:
        best_f1_epi = f1
        best_t3 = t3

print(f"  T1={best_t1:.2f}  T2={t2_base:.2f}  T3={best_t3:.2f}")

# ── Evaluar en test ───────────────────────────────────────────────────────────
print("[3/3] Evaluando en test 2021-2022...")
CLASES = ['Normal', 'Vigilancia', 'Alerta', 'Epidemia']
pc_test = clasificar(pred_test, best_t1, t2_base, best_t3)

acc   = accuracy_score(yc_test, pc_test)
kappa = cohen_kappa_score(yc_test, pc_test)

epi_r = (yc_test == 3); epi_p = (pc_test == 3)
tp = (epi_r & epi_p).sum(); fp = (~epi_r & epi_p).sum(); fn = (epi_r & ~epi_p).sum()
prec_epi = tp/(tp+fp) if (tp+fp)>0 else 0
rec_epi  = tp/(tp+fn) if (tp+fn)>0 else 0
f1_epi   = 2*prec_epi*rec_epi/(prec_epi+rec_epi) if (prec_epi+rec_epi)>0 else 0

print(f"\n{'='*60}")
print(f"  CLASIFICACIÓN CALIBRADA v2 — Test 2021-2022")
print(f"{'='*60}")
print(f"  {'':20} {'Antes':>10} {'Calibrado':>10}")
print(f"  {'Accuracy':20} {'57.2%':>10} {acc*100:>9.1f}%")
print(f"  {'Kappa':20} {'0.411':>10} {kappa:>10.3f}")
print(f"  {'Epidemia Recall':20} {'58.0%':>10} {rec_epi*100:>9.1f}%")
print(f"  {'Epidemia Precision':20} {'86.5%':>10} {prec_epi*100:>9.1f}%")
print(f"  {'Epidemia F1':20} {'69.5%':>10} {f1_epi*100:>9.1f}%")
print(f"  {'Falsos positivos':20} {'22':>10} {fp:>10}")
print(f"\n{classification_report(yc_test, pc_test, target_names=CLASES, digits=3)}")

# ── Guardar ───────────────────────────────────────────────────────────────────
out = {
    "t1_normal_vigilancia": round(float(best_t1), 4),
    "t2_vigilancia_alerta": round(float(t2_base),  4),
    "t3_alerta_epidemia":   round(float(best_t3),  4),
    "acc_test":   round(float(acc),       4),
    "kappa_test": round(float(kappa),     4),
    "epidemia_precision": round(float(prec_epi), 4),
    "epidemia_recall":    round(float(rec_epi),  4),
    "epidemia_f1":        round(float(f1_epi),   4),
}
out_path = os.path.join(MODELS, 'thresholds_clasificacion.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=4)
s3.upload(out_path, s3.PREFIX_MODELOS + 'thresholds_clasificacion.json')
print(f"  thresholds_clasificacion.json actualizado en S3.")
