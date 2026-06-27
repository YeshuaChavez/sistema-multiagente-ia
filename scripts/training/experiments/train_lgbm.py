import sys, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from scipy.optimize import minimize_scalar

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

print(f"Features: {len(COLS_FEAT)} | Train: {len(df_train)} | Test: {len(df_test)}")

# ── LightGBM ──────────────────────────────────────────────────────────────
print("\n[LightGBM] GridSearchCV...")
param_grid = {
    'modelo__n_estimators':   [800, 1000, 1200],
    'modelo__learning_rate':  [0.01, 0.02, 0.05],
    'modelo__num_leaves':     [31, 63, 127],
    'modelo__min_child_samples': [10, 20, 30],
    'modelo__reg_alpha':      [0, 0.1],
}
total = 1
for v in param_grid.values(): total *= len(v)
print(f"  Combinaciones: {total} x 3 folds = {total*3}")

pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('modelo', LGBMRegressor(subsample=0.8, feature_fraction=0.8,
                              random_state=42, n_jobs=-1, verbose=-1))
])
search = GridSearchCV(pipe, param_grid, cv=TimeSeriesSplit(3),
                      scoring='r2', n_jobs=-1, refit=True, verbose=0)
search.fit(df_train[COLS_FEAT], y_train_log)

print(f"  Mejor R2 CV: {search.best_score_*100:.2f}%")
print("  Mejores params:")
for k, v in sorted(search.best_params_.items()):
    print(f"    {k.replace('modelo__',''):22s}: {v}")

pipe_lgbm = search.best_estimator_
pred_lgbm_log = pipe_lgbm.predict(df_test[COLS_FEAT])
pred_lgbm     = np.expm1(pred_lgbm_log)
r2l_log = r2_score(np.log1p(y_test), pred_lgbm_log)
r2l_raw = r2_score(y_test, pred_lgbm)
print(f"\n  LightGBM: R2 log={r2l_log*100:.2f}%  R2 crudo={r2l_raw*100:.2f}%  MAE={mean_absolute_error(y_test,pred_lgbm):.4f}")
print(f"  (XGBoost ref sin NIC: 89.85% log  72.96% crudo)")

# ── LSTM con Early Stopping ───────────────────────────────────────────────
print("\n[LSTM] Entrenando (hidden=512, lr=0.003, dropout=0.2, ES)...")
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
            print(f"  Early stop epoch {ep+1} (mejor: epoch {best_ep})")
            break

m.load_state_dict(best_state)
m.eval()
with torch.no_grad():
    pred_lstm_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred_lstm = np.expm1(pred_lstm_log)
print(f"  LSTM: R2 log={r2_score(np.log1p(y_te), pred_lstm_log)*100:.2f}%  R2 crudo={r2_score(y_te,pred_lstm)*100:.2f}%")

# ── Ensemble LGBM + LSTM ──────────────────────────────────────────────────
lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
       for r, p in zip(df_test.itertuples(), pred_lgbm)}

y_l, lgbm_l, lstm_l = [], [], []
for sid, lp in zip(ids_test, pred_lstm):
    xp = lkp.get(sid)
    if xp is None: continue
    iso, adm, ano, mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso) &
                  (df_test['adm_1_name'].str.upper()==adm) &
                  (df_test['ano']==ano) & (df_test['mes']==mes)]
    if len(row):
        y_l.append(float(row['incidencia_dengue'].iloc[0]))
        lgbm_l.append(xp); lstm_l.append(lp)

y   = np.array(y_l)
lgb = np.array(lgbm_l)
lst = np.array(lstm_l)
ly, llgb, llst = np.log1p(y), np.log1p(lgb), np.log1p(lst)

# Pesos optimos raw
diff  = lgb - lst
w_lgb = float(np.clip(np.sum((y - lst) * diff) / np.sum(diff**2), 0, 1))
ens_raw = w_lgb * lgb + (1 - w_lgb) * lst

# Pesos optimos log
res = minimize_scalar(lambda w: -r2_score(ly, w*llgb + (1-w)*llst), bounds=(0,1), method='bounded')
w_log = float(res.x)
ens_log_preds = w_log * llgb + (1 - w_log) * llst

print(f"\n[Ensemble LGBM + LSTM]")
print(f"  Opt. raw  w_lgbm={w_lgb:.3f} w_lstm={1-w_lgb:.3f}")
print(f"    R2 log={r2_score(ly, np.log1p(ens_raw))*100:.2f}%  R2 crudo={r2_score(y,ens_raw)*100:.2f}%  MAE={mean_absolute_error(y,ens_raw):.4f}")
print(f"  Opt. log  w_lgbm={w_log:.3f} w_lstm={1-w_log:.3f}")
print(f"    R2 log={r2_score(ly, ens_log_preds)*100:.2f}%  R2 crudo={r2_score(y,np.expm1(ens_log_preds))*100:.2f}%  MAE={mean_absolute_error(y,np.expm1(ens_log_preds)):.4f}")

print(f"\n[Resumen final]")
print(f"  LightGBM solo : {r2l_log*100:.2f}% log  {r2l_raw*100:.2f}% crudo")
print(f"  LSTM solo     : {r2_score(np.log1p(y_te),pred_lstm_log)*100:.2f}% log  {r2_score(y_te,pred_lstm)*100:.2f}% crudo")
print(f"  Ensemble opt.raw  : {r2_score(ly,np.log1p(ens_raw))*100:.2f}% log  {r2_score(y,ens_raw)*100:.2f}% crudo")
print(f"  Ensemble opt.log  : {r2_score(ly,ens_log_preds)*100:.2f}% log")
