import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

COLS_EXCLUIR  = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT     = [c for c in df.columns if c not in COLS_EXCLUIR]
LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

df_test  = df[df['ano'] >= 2021].copy()
y_test   = df_test['incidencia_dengue'].values
print(f"Sin NIC: {len(df)} filas | {df['pais'].nunique()} paises")

def entrenar_xgb_pipe(df_tr, cols):
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('m',   XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                              min_child_weight=3, gamma=0.1, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
    ])
    pipe.fit(df_tr[cols], np.log1p(df_tr['incidencia_dengue']))
    return pipe

# ── Paso 1: Generar residuos OOF del XGBoost ─────────────────────────────
print("\n[Paso 1] Generando residuos out-of-fold para entrenamiento LSTM...")
folds_oof = [
    (df['ano'] <= 2017, df['ano'] == 2018),
    (df['ano'] <= 2018, df['ano'] == 2019),
    (df['ano'] <= 2019, df['ano'] == 2020),
]
df_oof_list = []
for tr_m, va_m in folds_oof:
    pipe_f = entrenar_xgb_pipe(df[tr_m], COLS_FEAT)
    df_va  = df[va_m].copy()
    df_va['xgb_pred_log']  = pipe_f.predict(df_va[COLS_FEAT])
    df_va['y_log']         = np.log1p(df_va['incidencia_dengue'])
    df_va['residuo_log']   = df_va['y_log'] - df_va['xgb_pred_log']
    df_oof_list.append(df_va)
    yr = df[va_m]['ano'].iloc[0]
    print(f"  Fold ano={yr}: R2 XGB={r2_score(df_va['y_log'], df_va['xgb_pred_log'])*100:.2f}%  "
          f"residuo std={df_va['residuo_log'].std():.4f}")

df_oof = pd.concat(df_oof_list)
print(f"  Total residuos OOF: {len(df_oof)} puntos (anos 2018-2020)")

# ── Paso 2: Construir secuencias para el LSTM de residuos ────────────────
print("\n[Paso 2] Construyendo secuencias LSTM sobre residuos...")
df_lstm_tr = df[df['ano'] <= 2020].copy()
df_lstm_tr = df_lstm_tr.merge(
    df_oof[['iso_a0','adm_1_name','ano','mes','residuo_log']],
    on=['iso_a0','adm_1_name','ano','mes'], how='left'
)
df_lstm_tr['residuo_log'] = df_lstm_tr['residuo_log'].fillna(0.0)

# Las secuencias usan features originales pero el target es el residuo
X_seq, _, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)

# Mapa de residuos
res_map = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(r.residuo_log)
           for r in df_oof.itertuples()}

# Solo entrenar en secuencias donde tenemos residuo (2018-2020)
tr_mask = (anos_seq >= 2018) & (anos_seq <= 2020)
te_mask = anos_seq >= 2021

X_tr_r = X_seq[tr_mask]
ids_tr  = [seq_ids[i] for i, m in enumerate(tr_mask) if m]
y_tr_r  = np.array([res_map.get(sid, 0.0) for sid in ids_tr], dtype=np.float32)

X_te_r  = X_seq[te_mask]
ids_test= [seq_ids[i] for i, m in enumerate(te_mask) if m]

print(f"  LSTM residuo — Train: {len(X_tr_r)} | Test: {len(X_te_r)}")
print(f"  Residuo train — mean={y_tr_r.mean():.4f}  std={y_tr_r.std():.4f}")

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr_r.reshape(-1,6)).reshape(X_tr_r.shape)
X_te_sc = sc.transform(X_te_r.reshape(-1,6)).reshape(X_te_r.shape)

# ── Paso 3: Entrenar LSTM de residuos ────────────────────────────────────
print("\n[Paso 3] Entrenando LSTM de residuos...")
torch.manual_seed(9); np.random.seed(9)
val_cut = int(len(X_tr_sc) * 0.85)
Xf = torch.tensor(X_tr_sc[:val_cut], dtype=torch.float32)
yf = torch.tensor(y_tr_r[:val_cut], dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_tr_sc[val_cut:], dtype=torch.float32)
yv = torch.tensor(y_tr_r[val_cut:],  dtype=torch.float32)

m   = DengueLSTMModel(6, 256, dropout=0.2)
opt = optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
loader = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
best_val, wait, best_state, best_ep = 1e9, 0, None, 0
for ep in range(300):
    m.train()
    for bx, by in loader:
        opt.zero_grad(); nn.MSELoss()(m(bx), by).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        vl = float(nn.MSELoss()(m(Xv).flatten(), yv))
    sched.step(vl)
    if vl < best_val - 1e-5:
        best_val = vl; wait = 0; best_ep = ep+1
        best_state = {k: v.clone() for k, v in m.state_dict().items()}
    else:
        wait += 1
        if wait >= 15:
            print(f"  Early stop epoch {ep+1} (mejor: epoch {best_ep})")
            break

m.load_state_dict(best_state); m.eval()
with torch.no_grad():
    residuo_pred = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
print(f"  Residuo predicho — mean={residuo_pred.mean():.4f}  std={residuo_pred.std():.4f}")

# ── Paso 4: XGBoost final + ensemble ────────────────────────────────────
print("\n[Paso 4] XGBoost final (≤2020) + ensemble...")
pipe_final = entrenar_xgb_pipe(df[df['ano'] <= 2020], COLS_FEAT)
pred_xgb_log = pipe_final.predict(df_test[COLS_FEAT])
pred_xgb     = np.expm1(pred_xgb_log)

# Alinear residuos con test
lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
       for r, p in zip(df_test.itertuples(), pred_xgb)}
y_l, xgb_l, res_l = [], [], []
for sid, rp in zip(ids_test, residuo_pred):
    xp = lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso)&(df_test['adm_1_name'].str.upper()==adm)&(df_test['ano']==ano)&(df_test['mes']==mes)]
    if len(row):
        y_l.append(float(row['incidencia_dengue'].iloc[0]))
        xgb_l.append(np.log1p(xp)); res_l.append(rp)

y, xgb_log_a, res_a = np.array(y_l), np.array(xgb_l), np.array(res_l)
ly = np.log1p(y)

ens_log = xgb_log_a + res_a
ens     = np.expm1(ens_log)

print(f"\n{'='*55}")
print(f"  RESIDUAL LSTM ENSEMBLE")
print(f"{'='*55}")
print(f"  XGBoost solo  : R2={r2_score(ly, xgb_log_a)*100:.2f}%")
print(f"  XGB + residuo : R2={r2_score(ly, ens_log)*100:.2f}%  MAE={mean_absolute_error(y,ens):.4f}")
print(f"  (referencia ensemble anterior: 87.17%)")
print(f"{'='*55}")
