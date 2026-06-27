# -*- coding: utf-8 -*-
"""
Compara XGBoost vs Ensemble por régimen epidémico en el test set 2021-2022.
Calcula R² y MAE segmentados por nivel de incidencia real:
  Normal     : real <= p25 local del departamento
  Vigilancia : p25 < real <= p50
  Alerta     : p50 < real <= p90
  Brote      : real > p90  ← el que más importa

Uso: python analizar_por_regimen.py
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODELS = os.path.join(BASE, 'Base de Datos', 'modelos')
PROC   = os.path.join(BASE, 'Base de Datos', 'datos_procesados')

# ── Cargar datos ──────────────────────────────────────────────────────────────
df       = pd.read_csv(os.path.join(PROC, 'dataset_features_latam.csv'))
df_master= pd.read_csv(os.path.join(PROC, 'dataset_maestro_mensual_latam.csv'))

with open(os.path.join(MODELS, 'cols_feat.pkl'), 'rb') as f:
    cols_feat = pickle.load(f)
with open(os.path.join(MODELS, 'metrics.json')) as f:
    metrics = json.load(f)

w_xgb  = metrics.get('ensemble_w_xgb',  0.90)
w_lstm = metrics.get('ensemble_w_lstm', 0.10)

# ── XGBoost ───────────────────────────────────────────────────────────────────
with open(os.path.join(MODELS, 'pipeline_ml.pkl'), 'rb') as f:
    pipe_xgb = pickle.load(f)

# ── LSTM ──────────────────────────────────────────────────────────────────────
with open(os.path.join(MODELS, 'lstm_config.json')) as f:
    lstm_cfg = json.load(f)
with open(os.path.join(MODELS, 'lstm_features.pkl'), 'rb') as f:
    lstm_feats = pickle.load(f)
with open(os.path.join(MODELS, 'escalador_lstm.pkl'), 'rb') as f:
    scaler_lstm = pickle.load(f)

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

# ── Predicciones test set completo ────────────────────────────────────────────
df_test   = df[df['ano'].isin([2021, 2022])].copy()
y_real    = df_test['incidencia_dengue'].values
pred_xgb  = np.expm1(pipe_xgb.predict(df_test[cols_feat]))

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
valid = ~np.isnan(pred_lstm)
pred_ens = w_xgb * pred_xgb + w_lstm * np.where(valid, pred_lstm, pred_xgb)

# ── Calcular p25/p50/p90 LOCAL por departamento (solo datos train 2014-2020) ──
df_hist = df_master[df_master['ano'] <= 2020].copy()
df_hist['iso_u'] = df_hist['iso_a0'].str.strip().str.upper()
df_hist['adm_u'] = df_hist['adm_1_name'].str.strip().str.upper()

pct_map = {}
for (iso, adm), grp in df_hist.groupby(['iso_u', 'adm_u']):
    inc = grp['incidencia_dengue']
    pct_map[(iso, adm)] = (
        float(inc.quantile(0.25)),
        float(inc.quantile(0.50)),
        float(inc.quantile(0.90)),
    )

# Fallback global (train)
g25 = float(df_hist['incidencia_dengue'].quantile(0.25))
g50 = float(df_hist['incidencia_dengue'].quantile(0.50))
g90 = float(df_hist['incidencia_dengue'].quantile(0.90))

# ── Asignar régimen a cada fila del test ──────────────────────────────────────
regimenes = []
for row in df_test.itertuples():
    iso = str(row.iso_a0).strip().upper()
    adm = str(row.adm_1_name).strip().upper()
    p25, p50, p90 = pct_map.get((iso, adm), (g25, g50, g90))
    p90 = max(p90, g90)  # floor global igual que Agente 6
    inc = row.incidencia_dengue
    if inc <= p25:
        regimenes.append('Normal')
    elif inc <= p50:
        regimenes.append('Vigilancia')
    elif inc <= p90:
        regimenes.append('Alerta')
    else:
        regimenes.append('Brote')

regimenes = np.array(regimenes)

# ── Métricas por régimen ──────────────────────────────────────────────────────
def metricas(y, px, pe, mask):
    if mask.sum() < 5:
        return None
    y_m  = y[mask];  px_m = px[mask];  pe_m = pe[mask]
    ly_m = np.log1p(y_m)
    return {
        'n':        int(mask.sum()),
        'pct':      round(mask.sum() / len(y) * 100, 1),
        'r2_xgb':   round(r2_score(ly_m, np.log1p(px_m)) * 100, 2),
        'r2_ens':   round(r2_score(ly_m, np.log1p(pe_m)) * 100, 2),
        'mae_xgb':  round(mean_absolute_error(y_m, px_m), 2),
        'mae_ens':  round(mean_absolute_error(y_m, pe_m), 2),
    }

regiones = ['Normal', 'Vigilancia', 'Alerta', 'Brote']
resultados = {}
for reg in regiones:
    mask = regimenes == reg
    resultados[reg] = metricas(y_real, pred_xgb, pred_ens, mask)

# ── Imprimir tabla ────────────────────────────────────────────────────────────
print(f"\nPesos ensemble: w_xgb={w_xgb:.2f}  w_lstm={w_lstm:.2f}")
print(f"\n{'Régimen':<12} {'N':>5} {'%test':>6}  {'R² XGB':>8} {'R² Ens':>8} {'Δ R²':>7}  {'MAE XGB':>8} {'MAE Ens':>8} {'Δ MAE':>7}")
print("─" * 80)
for reg in regiones:
    r = resultados[reg]
    if r is None:
        continue
    delta_r2  = r['r2_ens']  - r['r2_xgb']
    delta_mae = r['mae_ens'] - r['mae_xgb']
    signo_r2  = "+" if delta_r2  >= 0 else ""
    signo_mae = "+" if delta_mae >= 0 else ""
    print(f"{reg:<12} {r['n']:>5} {r['pct']:>5.1f}%  "
          f"{r['r2_xgb']:>7.2f}% {r['r2_ens']:>7.2f}% {signo_r2}{delta_r2:>6.2f}%  "
          f"{r['mae_xgb']:>8.2f} {r['mae_ens']:>8.2f} {signo_mae}{delta_mae:>6.2f}")

print("─" * 80)
mask_all = np.ones(len(y_real), dtype=bool)
r_all = metricas(y_real, pred_xgb, pred_ens, mask_all)
delta_r2  = r_all['r2_ens'] - r_all['r2_xgb']
delta_mae = r_all['mae_ens'] - r_all['mae_xgb']
print(f"{'TOTAL':<12} {r_all['n']:>5} {'100.0':>5}%  "
      f"{r_all['r2_xgb']:>7.2f}% {r_all['r2_ens']:>7.2f}% {'+' if delta_r2>=0 else ''}{delta_r2:>6.2f}%  "
      f"{r_all['mae_xgb']:>8.2f} {r_all['mae_ens']:>8.2f} {'+' if delta_mae>=0 else ''}{delta_mae:>6.2f}")

brote = resultados['Brote']
print(f"\n>>> En meses de BROTE: Ensemble {'GANA' if brote['r2_ens'] > brote['r2_xgb'] else 'PIERDE'} "
      f"({brote['r2_ens']:.2f}% vs {brote['r2_xgb']:.2f}% en R², "
      f"MAE {brote['mae_ens']:.2f} vs {brote['mae_xgb']:.2f})")
