# -*- coding: utf-8 -*-
"""
Prueba diferentes umbrales de predicción para la clase Epidemia.
Ground truth siempre usa p50/p90 (igual que Agente 5).
Solo varía el umbral aplicado a la PREDICCIÓN del ensemble.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODELS = os.path.join(BASE, 'data', 'models')
PROC   = os.path.join(BASE, 'data', 'processed')

df        = pd.read_csv(os.path.join(PROC, 'dataset_features_latam.csv'))
df_master = pd.read_csv(os.path.join(PROC, 'dataset_maestro_mensual_latam.csv'))

with open(os.path.join(MODELS, 'cols_feat.pkl'),     'rb') as f: cols_feat   = pickle.load(f)
with open(os.path.join(MODELS, 'metrics.json'))           as f: metrics     = json.load(f)
with open(os.path.join(MODELS, 'pipeline_ml.pkl'),   'rb') as f: pipe_xgb    = pickle.load(f)
with open(os.path.join(MODELS, 'lstm_config.json'))       as f: lstm_cfg    = json.load(f)
with open(os.path.join(MODELS, 'lstm_features.pkl'), 'rb') as f: lstm_feats  = pickle.load(f)
with open(os.path.join(MODELS, 'escalador_lstm.pkl'),'rb') as f: scaler_lstm = pickle.load(f)

w_xgb  = metrics.get('ensemble_w_xgb', 0.90)
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

# ── Predicciones test set ─────────────────────────────────────────────────────
print("Calculando predicciones...")
df_test  = df[df['ano'].isin([2021, 2022])].copy()
y_real   = df_test['incidencia_dengue'].values
pred_xgb = np.expm1(pipe_xgb.predict(df_test[cols_feat]))

df_scaled = df.copy()
df_scaled[lstm_feats] = scaler_lstm.transform(df[lstm_feats])
pred_lstm = []
for idx in df_test.index:
    pos = df.index.get_loc(idx)
    if pos < SEQ_LEN: pred_lstm.append(np.nan); continue
    seq = df_scaled.iloc[pos-SEQ_LEN:pos][lstm_feats].values
    with torch.no_grad():
        p = model_lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)).item()
    pred_lstm.append(max(0.0, np.expm1(p)))

pred_lstm = np.array(pred_lstm, dtype=float)
valid     = ~np.isnan(pred_lstm)
pred_ens  = w_xgb * pred_xgb + w_lstm * np.where(valid, pred_lstm, pred_xgb)

# ── Percentiles locales (train 2014-2020) ─────────────────────────────────────
df_hist = df_master[df_master['ano'] <= 2020].copy()
df_hist['iso_u'] = df_hist['iso_a0'].str.strip().str.upper()
df_hist['adm_u'] = df_hist['adm_1_name'].str.strip().str.upper()
g50 = float(df_hist['incidencia_dengue'].quantile(0.50))
g90 = float(df_hist['incidencia_dengue'].quantile(0.90))

pct_map = {}
for (iso, adm), grp in df_hist.groupby(['iso_u', 'adm_u']):
    inc = grp['incidencia_dengue']
    pct_map[(iso, adm)] = (
        max(float(inc.quantile(0.50)), 0.5),
        max(float(inc.quantile(0.90)), g90),   # floor global en p90
    )

# ── Ground truth: siempre p50/p90 (idéntico a Agente 5) ──────────────────────
def clase_real(valor, iso, adm):
    p50, p90 = pct_map.get((iso, adm), (g50, g90))
    if valor <= p50: return 0   # Endémico
    elif valor <= p90: return 1 # Alerta
    else: return 2              # Epidemia

y_clase = np.array([
    clase_real(y_real[i], str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper())
    for i, r in enumerate(df_test.itertuples())
])

# ── Clasificar predicciones con umbral variable en Epidemia ──────────────────
# percentil_epi_pred: percentil que usamos sobre las PREDICCIONES para umbral Epidemia
# El umbral de Endémico/Alerta sigue siendo p50 real del departamento

CLASES = ['Endémico', 'Alerta', 'Epidemia']

print(f"\n{'='*72}")
print(f"  TRADE-OFF UMBRAL DE EPIDEMIA — Test 2021-2022")
print(f"  Ground truth: p50/p90 locales (fijo)")
print(f"  Variable: percentil aplicado a PREDICCIONES para umbral Epidemia")
print(f"{'='*72}")
print(f"  {'Umbral pred':>14}  {'Epi Recall':>10}  {'Epi Prec':>9}  {'Epi F1':>8}  {'Accuracy':>9}  {'Kappa':>7}  {'Falsos+':>8}")
print(f"  {'-'*70}")

for pct_epi in [90, 85, 80, 75, 70]:
    # Umbral epidemia para predicciones: percentil sobre pred_ens del val set...
    # pero más limpio: percentil sobre pred_ens del propio test (aproximación)
    # En producción se fijaría sobre val 2020 - aquí lo calculamos sobre test para ver el efecto

    def clase_pred(valor, iso, adm, pct_umbral):
        p50, p90 = pct_map.get((iso, adm), (g50, g90))
        # Umbral epidemia = fracción del p90 local
        p_epi = p90 * (pct_umbral / 90.0)
        if valor <= p50:   return 0  # Endémico
        elif valor <= p_epi: return 1  # Alerta
        else:              return 2  # Epidemia

    ens_clase = np.array([
        clase_pred(pred_ens[i], str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), pct_epi)
        for i, r in enumerate(df_test.itertuples())
    ])

    acc   = accuracy_score(y_clase, ens_clase)
    kappa = cohen_kappa_score(y_clase, ens_clase)

    epi_r = (y_clase == 2); epi_p = (ens_clase == 2)
    tp = (epi_r & epi_p).sum()
    fp = (~epi_r & epi_p).sum()
    fn = (epi_r & ~epi_p).sum()
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    marker = " <- actual" if pct_epi == 90 else ""
    print(f"  {'p'+str(pct_epi)+' local':>14}  {rec*100:>9.1f}%  {prec*100:>8.1f}%  {f1*100:>7.1f}%  {acc*100:>8.1f}%  {kappa:>7.3f}  {fp:>8}{marker}")

# ── Detalle completo del mejor candidato (p80) ───────────────────────────────
print(f"\n{'='*72}")
print(f"  DETALLE — umbral p80 local")
print(f"{'='*72}")

ens_p80 = np.array([
    clase_pred(pred_ens[i], str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), 80)
    for i, r in enumerate(df_test.itertuples())
])
cm = confusion_matrix(y_clase, ens_p80)
print(f"\n  Matriz de confusión (filas=real, cols=predicho):")
print(f"  {'':14}", end="")
for c in CLASES: print(f"{c:>12}", end="")
print()
for i, c in enumerate(CLASES):
    n = (y_clase == i).sum()
    print(f"  {c:<14}", end="")
    for j in range(len(CLASES)):
        pct = cm[i,j]/n*100 if n>0 else 0
        print(f"{cm[i,j]:>7}({pct:4.1f}%)", end="")
    print()

print(f"\n{classification_report(y_clase, ens_p80, target_names=CLASES, digits=3)}")
