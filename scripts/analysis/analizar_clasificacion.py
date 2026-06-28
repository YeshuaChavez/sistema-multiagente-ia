# -*- coding: utf-8 -*-
"""
Evalúa qué tan bien clasifica el sistema los niveles de riesgo epidémico
(Endémico / Alerta / Epidemia) sobre el test set 2021-2022.
Lógica idéntica a Agente 5: pred <= p50 → Endémico, <= p90 → Alerta, > p90 → Epidemia.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, cohen_kappa_score)
sys.path.insert(0, 'agents')
from dotenv import load_dotenv; load_dotenv('.env')

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODELS = os.path.join(BASE, 'data', 'models')
PROC   = os.path.join(BASE, 'data', 'processed')

df       = pd.read_csv(os.path.join(PROC, 'dataset_features_latam.csv'))
df_master= pd.read_csv(os.path.join(PROC, 'dataset_maestro_mensual_latam.csv'))

with open(os.path.join(MODELS, 'cols_feat.pkl'), 'rb') as f:
    cols_feat = pickle.load(f)
with open(os.path.join(MODELS, 'metrics.json')) as f:
    metrics = json.load(f)
with open(os.path.join(MODELS, 'pipeline_ml.pkl'), 'rb') as f:
    pipe_xgb = pickle.load(f)
with open(os.path.join(MODELS, 'lstm_config.json')) as f:
    lstm_cfg = json.load(f)
with open(os.path.join(MODELS, 'lstm_features.pkl'), 'rb') as f:
    lstm_feats = pickle.load(f)
with open(os.path.join(MODELS, 'escalador_lstm.pkl'), 'rb') as f:
    scaler_lstm = pickle.load(f)

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

# ── Predicciones ──────────────────────────────────────────────────────────────
df_test  = df[df['ano'].isin([2021, 2022])].copy()
y_real   = df_test['incidencia_dengue'].values
pred_xgb = np.expm1(pipe_xgb.predict(df_test[cols_feat]))

df_scaled = df.copy()
df_scaled[lstm_feats] = scaler_lstm.transform(df[lstm_feats])
pred_lstm = []
for idx in df_test.index:
    pos = df.index.get_loc(idx)
    if pos < SEQ_LEN:
        pred_lstm.append(np.nan)
        continue
    seq = df_scaled.iloc[pos - SEQ_LEN:pos][lstm_feats].values
    with torch.no_grad():
        p = model_lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)).item()
    pred_lstm.append(max(0.0, np.expm1(p)))

pred_lstm = np.array(pred_lstm, dtype=float)
valid     = ~np.isnan(pred_lstm)
pred_ens  = w_xgb * pred_xgb + w_lstm * np.where(valid, pred_lstm, pred_xgb)

# ── Percentiles locales por departamento (entrenamiento 2014-2020) ────────────
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

def clasificar(valor, iso, adm):
    _, p50, p90 = pct_map.get((iso, adm), (g25, g50, g90))
    if valor <= p50:   return 0  # Endémico
    elif valor <= p90: return 1  # Alerta
    else:              return 2  # Epidemia

CLASES = ['Endémico', 'Alerta', 'Epidemia']

y_clase, ens_clase = [], []
for i, row in enumerate(df_test.itertuples()):
    iso = str(row.iso_a0).strip().upper()
    adm = str(row.adm_1_name).strip().upper()
    y_clase.append(clasificar(y_real[i], iso, adm))
    ens_clase.append(clasificar(pred_ens[i], iso, adm))

y_clase   = np.array(y_clase)
ens_clase = np.array(ens_clase)

# ── Resultados ────────────────────────────────────────────────────────────────
acc   = accuracy_score(y_clase, ens_clase)
kappa = cohen_kappa_score(y_clase, ens_clase)
cm    = confusion_matrix(y_clase, ens_clase)

print(f"\n{'='*55}")
print(f"  CLASIFICACIÓN DE NIVEL DE RIESGO — Test 2021-2022")
print(f"{'='*55}")
print(f"  Accuracy : {acc*100:.1f}%")
print(f"  Kappa    : {kappa:.3f}  ", end="")
if kappa >= 0.8:   print("(Casi perfecto)")
elif kappa >= 0.6: print("(Sustancial)")
elif kappa >= 0.4: print("(Moderado)")
else:              print("(Débil)")

print(f"\n  Matriz de confusión (filas=real, cols=predicho):")
print(f"  {'':12}", end="")
for c in CLASES: print(f"{c:>12}", end="")
print()
for i, c in enumerate(CLASES):
    n_real = (y_clase == i).sum()
    print(f"  {c:<12}", end="")
    for j in range(len(CLASES)):
        val = cm[i, j]
        pct = val / n_real * 100 if n_real > 0 else 0
        print(f"{val:>7}({pct:4.1f}%)", end="")
    print()

print(f"\n  Reporte por clase:")
print(classification_report(y_clase, ens_clase, target_names=CLASES, digits=3))

# ── Métrica crítica: detección de epidemias ───────────────────────────────────
epi_real  = (y_clase == 2)
epi_pred  = (ens_clase == 2)
tp = (epi_real & epi_pred).sum()
fn = (epi_real & ~epi_pred).sum()
fp = (~epi_real & epi_pred).sum()
recall_epi    = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_epi = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_epi        = 2 * precision_epi * recall_epi / (precision_epi + recall_epi) if (precision_epi + recall_epi) > 0 else 0

print(f"  EPIDEMIA — detección crítica:")
print(f"    Recall    : {recall_epi*100:.1f}%  (de cada 100 brotes reales, detecta {recall_epi*100:.0f})")
print(f"    Precision : {precision_epi*100:.1f}%  (de cada 100 alarmas, {precision_epi*100:.0f} son reales)")
print(f"    F1-score  : {f1_epi*100:.1f}%")
print(f"    Brotes reales: {epi_real.sum()} | Detectados: {tp} | Perdidos: {fn} | Falsos: {fp}")
