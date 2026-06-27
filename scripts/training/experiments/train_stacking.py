import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
from agente_4_prediccion_dl import DengueLSTMModel, _build_sequences, LSTM_SEQ_LEN

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
stats = df.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok = stats[stats <= 1000].index
df = df[df.set_index(['pais','adm_1_name']).index.isin(deptos_ok)].reset_index(drop=True)
print(f"Dataset: {len(df)} filas | {df['pais'].nunique()} paises")

COLS_EXCLUIR  = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT     = [c for c in df.columns if c not in COLS_EXCLUIR]
LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

# Splits: base=≤2018, meta=2019-2020, test=2021+
df_base = df[df['ano'] <= 2018]
df_meta = df[(df['ano'] >= 2019) & (df['ano'] <= 2020)].copy()
df_test = df[df['ano'] >= 2021].copy()
df_train_full = df[df['ano'] <= 2020]
print(f"Base (≤2018): {len(df_base)} | Meta (2019-20): {len(df_meta)} | Test (2021+): {len(df_test)}")

# ── Nivel 1A: XGBoost ──────────────────────────────────────────────────────
def entrenar_xgb(df_tr, df_te, cols):
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('m',   XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                              min_child_weight=3, gamma=0.1, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
    ])
    pipe.fit(df_tr[cols], np.log1p(df_tr['incidencia_dengue']))
    return np.expm1(pipe.predict(df_te[cols])), pipe

print("\n[Nivel 1] Entrenando modelos base en ≤2018...")
pred_xgb_meta, pipe_xgb_base = entrenar_xgb(df_base, df_meta, COLS_FEAT)

# ── Nivel 1B: LSTM ─────────────────────────────────────────────────────────
def entrenar_lstm_es(df_data, hidden=512, lr=0.003, dr=0.2, seed=9):
    X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df_data, LSTM_FEATURES, LSTM_SEQ_LEN)
    tr_m = anos_seq <= 2018
    te_m = (anos_seq >= 2019) & (anos_seq <= 2020)
    X_tr, y_tr = X_seq[tr_m], y_seq[tr_m]
    X_te, y_te = X_seq[te_m], y_seq[te_m]
    anos_tr = anos_seq[tr_m]
    ids_te = [seq_ids[i] for i, m in enumerate(te_m) if m]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr.reshape(-1,6)).reshape(X_tr.shape)
    X_te_sc = sc.transform(X_te.reshape(-1,6)).reshape(X_te.shape)
    y_tr_log = np.log1p(y_tr)

    torch.manual_seed(seed); np.random.seed(seed)
    val_m = anos_tr == 2018; fit_m = ~val_m
    Xf = torch.tensor(X_tr_sc[fit_m], dtype=torch.float32)
    yf = torch.tensor(y_tr_log[fit_m], dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_tr_sc[val_m], dtype=torch.float32)
    yv = torch.tensor(y_tr_log[val_m], dtype=torch.float32)

    m   = DengueLSTMModel(6, hidden, dropout=dr)
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl  = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
    best_val, wait, best_state = 1e9, 0, None
    for ep in range(300):
        m.train()
        for bx, by in dl:
            opt.zero_grad(); nn.MSELoss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(nn.MSELoss()(m(Xv).flatten(), yv))
        sch.step(vl)
        if vl < best_val - 1e-5:
            best_val = vl; wait = 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 15: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pred_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
    return np.expm1(pred_log), ids_te, y_te, sc, m

pred_lstm_meta, ids_meta, y_meta_lstm, sc_lstm_base, lstm_base = entrenar_lstm_es(df)
print(f"  XGBoost meta: {len(pred_xgb_meta)} preds | LSTM meta: {len(pred_lstm_meta)} preds")

# Alinear meta-features
lkp_meta = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): (float(px), float(r.incidencia_dengue))
            for r, px in zip(df_meta.itertuples(), pred_xgb_meta)}
y_m, xgb_m, lstm_m = [], [], []
for sid, lp in zip(ids_meta, pred_lstm_meta):
    v = lkp_meta.get(sid)
    if v is None: continue
    y_m.append(v[1]); xgb_m.append(v[0]); lstm_m.append(lp)

y_m, xgb_m, lstm_m = np.array(y_m), np.array(xgb_m), np.array(lstm_m)
print(f"  Meta alineados: {len(y_m)} puntos")

# ── Nivel 2: Meta-learner ──────────────────────────────────────────────────
# Features: pred_xgb, pred_lstm, log(pred_xgb), log(pred_lstm), pred_xgb*w + pred_lstm*(1-w)
def meta_features(xgb, lstm):
    return np.column_stack([
        np.log1p(xgb), np.log1p(lstm),
        xgb, lstm,
        np.log1p(0.5*xgb + 0.5*lstm),
    ])

X_meta_tr = meta_features(xgb_m, lstm_m)
y_meta_tr  = np.log1p(y_m)

print("\n[Nivel 2] Entrenando meta-learner Ridge...")
from sklearn.preprocessing import StandardScaler as SS
sc_meta = SS()
X_meta_sc = sc_meta.fit_transform(X_meta_tr)
ridge = Ridge(alpha=1.0)
ridge.fit(X_meta_sc, y_meta_tr)
print(f"  Ridge coefs: {ridge.coef_.round(4)}")

# ── Nivel 1 FINAL: reentrenar en todo ≤2020 ────────────────────────────────
print("\n[Final] Reentrenando modelos en ≤2020...")
pred_xgb_test, _ = entrenar_xgb(df_train_full, df_test, COLS_FEAT)

# LSTM final en todo ≤2020
X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)
tr_m = anos_seq <= 2020; te_m = anos_seq >= 2021
X_tr, y_tr = X_seq[tr_m], y_seq[tr_m]
X_te, y_te = X_seq[te_m], y_seq[te_m]
anos_tr = anos_seq[tr_m]; ids_test = [seq_ids[i] for i, m in enumerate(te_m) if m]

sc_f = StandardScaler()
X_tr_sc = sc_f.fit_transform(X_tr.reshape(-1,6)).reshape(X_tr.shape)
X_te_sc = sc_f.transform(X_te.reshape(-1,6)).reshape(X_te.shape)
y_tr_log = np.log1p(y_tr)
torch.manual_seed(9); np.random.seed(9)
val_m2 = anos_tr == 2020; fit_m2 = ~val_m2
Xf = torch.tensor(X_tr_sc[fit_m2], dtype=torch.float32)
yf = torch.tensor(y_tr_log[fit_m2], dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_tr_sc[val_m2], dtype=torch.float32)
yv = torch.tensor(y_tr_log[val_m2], dtype=torch.float32)
lstm_final = DengueLSTMModel(6, 512, dropout=0.2)
opt2 = optim.Adam(lstm_final.parameters(), lr=0.003, weight_decay=1e-4)
sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=5, factor=0.5)
dl2  = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
best_val2, wait2, best_state2 = 1e9, 0, None
for ep in range(300):
    lstm_final.train()
    for bx, by in dl2:
        opt2.zero_grad(); nn.MSELoss()(lstm_final(bx), by).backward(); opt2.step()
    lstm_final.eval()
    with torch.no_grad():
        vl = float(nn.MSELoss()(lstm_final(Xv).flatten(), yv))
    sch2.step(vl)
    if vl < best_val2 - 1e-5:
        best_val2 = vl; wait2 = 0
        best_state2 = {k: v.clone() for k, v in lstm_final.state_dict().items()}
    else:
        wait2 += 1
        if wait2 >= 15: break
lstm_final.load_state_dict(best_state2); lstm_final.eval()
with torch.no_grad():
    pred_lstm_log = lstm_final(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred_lstm_test = np.expm1(pred_lstm_log)

# Alinear test
lkp_test = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
            for r, p in zip(df_test.itertuples(), pred_xgb_test)}
y_t, xgb_t, lstm_t = [], [], []
for sid, lp in zip(ids_test, pred_lstm_test):
    xp = lkp_test.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso)&(df_test['adm_1_name'].str.upper()==adm)&(df_test['ano']==ano)&(df_test['mes']==mes)]
    if len(row):
        y_t.append(float(row['incidencia_dengue'].iloc[0]))
        xgb_t.append(xp); lstm_t.append(lp)

y_t, xgb_t, lstm_t = np.array(y_t), np.array(xgb_t), np.array(lstm_t)
ly, lxgb, llst = np.log1p(y_t), np.log1p(xgb_t), np.log1p(lstm_t)

# Meta-learner prediccion
X_test_meta = sc_meta.transform(meta_features(xgb_t, lstm_t))
pred_meta_log = ridge.predict(X_test_meta)
pred_meta = np.expm1(pred_meta_log)

print(f"\n{'='*55}")
print(f"  RESULTADOS FINALES (sin NIC, sin outliers extremos)")
print(f"{'='*55}")
print(f"  XGBoost solo  : R2 log={r2_score(ly,lxgb)*100:.2f}%  R2 crudo={r2_score(y_t,xgb_t)*100:.2f}%")
print(f"  LSTM solo     : R2 log={r2_score(ly,llst)*100:.2f}%  R2 crudo={r2_score(y_t,lstm_t)*100:.2f}%")
print(f"  Ensemble prom : R2 log={r2_score(ly,np.log1p(0.87*xgb_t+0.13*lstm_t))*100:.2f}%  R2 crudo={r2_score(y_t,0.87*xgb_t+0.13*lstm_t)*100:.2f}%")
print(f"  Meta-learner  : R2 log={r2_score(ly,pred_meta_log)*100:.2f}%  R2 crudo={r2_score(y_t,pred_meta)*100:.2f}%  MAE={mean_absolute_error(y_t,pred_meta):.4f}")
print(f"{'='*55}")
