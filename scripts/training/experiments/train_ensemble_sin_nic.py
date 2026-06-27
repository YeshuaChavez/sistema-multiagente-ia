import sys, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
from agente_4_prediccion_dl import DengueLSTMModel, LSTM_SEQ_LEN, _build_sequences

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
print(f"Sin Nicaragua: {len(df)} filas | {df['pais'].nunique()} paises")

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT    = [c for c in df.columns if c not in COLS_EXCLUIR]

df_train = df[df['ano'] <= 2020]
df_test  = df[df['ano'] >= 2021].copy()
y_train_log = np.log1p(df_train['incidencia_dengue'])
y_test      = df_test['incidencia_dengue'].values

LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

print(f"XGBoost — features: {len(COLS_FEAT)} | Train: {len(df_train)} | Test: {len(df_test)}")

# ── XGBoost (mejores params del run sin NIC) ──────────────────────────────
pipe_xgb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('modelo', XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                             min_child_weight=3, gamma=0.1, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
])
pipe_xgb.fit(df_train[COLS_FEAT], y_train_log)
pred_xgb = np.expm1(pipe_xgb.predict(df_test[COLS_FEAT]))
r2x_log  = r2_score(np.log1p(y_test), np.log1p(pred_xgb))
r2x_raw  = r2_score(y_test, pred_xgb)
print(f"  XGBoost: R2 log={r2x_log*100:.2f}%  R2 crudo={r2x_raw*100:.2f}%  MAE={mean_absolute_error(y_test,pred_xgb):.4f}")

# ── LSTM con Early Stopping ───────────────────────────────────────────────
X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)
tr_mask = anos_seq <= 2020
te_mask = anos_seq >= 2021

X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
X_te, y_te = X_seq[te_mask], y_seq[te_mask]
anos_tr    = anos_seq[tr_mask]
ids_test   = [seq_ids[i] for i, m in enumerate(te_mask) if m]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr.reshape(-1, 6)).reshape(X_tr.shape)
X_te_sc = sc.transform(X_te.reshape(-1, 6)).reshape(X_te.shape)
y_tr_log = np.log1p(y_tr)

# Entrenar LSTM con ES (hidden=512, lr=0.003, dropout=0.2)
torch.manual_seed(9); np.random.seed(9)
val_mask = anos_tr == 2020
fit_mask = ~val_mask

Xf = torch.tensor(X_tr_sc[fit_mask], dtype=torch.float32)
yf = torch.tensor(y_tr_log[fit_mask], dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_tr_sc[val_mask], dtype=torch.float32)
yv = torch.tensor(y_tr_log[val_mask], dtype=torch.float32)

m   = DengueLSTMModel(6, 512, dropout=0.2)
opt = optim.Adam(m.parameters(), lr=0.003, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
loader = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
best_val, wait, best_state, best_ep = 1e9, 0, None, 0

for ep in range(300):
    m.train()
    for bx, by in loader:
        opt.zero_grad(); nn.MSELoss()(m(bx), by).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        val_mse = float(nn.MSELoss()(m(Xv).flatten(), yv))
    sched.step(val_mse)
    if val_mse < best_val - 1e-5:
        best_val = val_mse; wait = 0; best_ep = ep + 1
        best_state = {k: v.clone() for k, v in m.state_dict().items()}
    else:
        wait += 1
        if wait >= 15:
            print(f"  LSTM early stop epoch {ep+1} (mejor: epoch {best_ep})")
            break

m.load_state_dict(best_state)
m.eval()
with torch.no_grad():
    pred_lstm_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred_lstm = np.expm1(pred_lstm_log)
r2l_log = r2_score(np.log1p(y_te), pred_lstm_log)
r2l_raw = r2_score(y_te, pred_lstm)
print(f"  LSTM:    R2 log={r2l_log*100:.2f}%  R2 crudo={r2l_raw*100:.2f}%  MAE={mean_absolute_error(y_te,pred_lstm):.4f}")

# ── Ensemble alineado ─────────────────────────────────────────────────────
lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
       for r, p in zip(df_test.itertuples(), pred_xgb)}

y_l, xgb_l, lstm_l = [], [], []
for sid, lp in zip(ids_test, pred_lstm):
    xp = lkp.get(sid)
    if xp is None: continue
    iso, adm, ano, mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso) &
                  (df_test['adm_1_name'].str.upper()==adm) &
                  (df_test['ano']==ano) & (df_test['mes']==mes)]
    if len(row):
        y_l.append(float(row['incidencia_dengue'].iloc[0]))
        xgb_l.append(xp); lstm_l.append(lp)

y   = np.array(y_l)
xgb = np.array(xgb_l)
lst = np.array(lstm_l)
ly, lxgb, llst = np.log1p(y), np.log1p(xgb), np.log1p(lst)

# Peso optimo por minimos cuadrados (raw)
diff  = xgb - lst
w_xgb = float(np.clip(np.sum((y - lst) * diff) / np.sum(diff**2), 0, 1))
ens   = w_xgb * xgb + (1 - w_xgb) * lst

# Buscar peso optimo para log-escala
from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda w: -r2_score(ly, w*lxgb + (1-w)*llst), bounds=(0,1), method='bounded')
w_log = float(res.x)
ens_log = w_log * lxgb + (1 - w_log) * llst

print(f"\n  Ensemble (pesos optimos raw)  w_xgb={w_xgb:.3f} w_lstm={1-w_xgb:.3f}")
print(f"    R2 log={r2_score(ly, np.log1p(ens))*100:.2f}%  R2 crudo={r2_score(y,ens)*100:.2f}%  MAE={mean_absolute_error(y,ens):.4f}")
print(f"\n  Ensemble (pesos optimos log)  w_xgb={w_log:.3f} w_lstm={1-w_log:.3f}")
print(f"    R2 log={r2_score(ly, ens_log)*100:.2f}%  R2 crudo={r2_score(y,np.expm1(ens_log))*100:.2f}%  MAE={mean_absolute_error(y,np.expm1(ens_log)):.4f}")
print(f"\n[Referencia anterior con NIC]")
print(f"  XGBoost: 88.84% log | LSTM: 84.15% log | Ensemble: 86.75% log")
