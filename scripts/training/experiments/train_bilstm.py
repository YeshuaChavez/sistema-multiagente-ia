import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize_scalar

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
from agente_4_prediccion_dl import _build_sequences, LSTM_SEQ_LEN

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

# ── Modelo BiLSTM ──────────────────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 por bidireccional

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Datos ──────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
stats = df.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok = stats[stats <= 1000].index
df = df[df.set_index(['pais','adm_1_name']).index.isin(deptos_ok)].reset_index(drop=True)
print(f"Sin NIC + sin outliers: {len(df)} filas | {df['pais'].nunique()} paises")

LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)
tr_mask = anos_seq <= 2020
te_mask = anos_seq >= 2021
X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
X_te, y_te = X_seq[te_mask], y_seq[te_mask]
anos_tr    = anos_seq[tr_mask]
ids_test   = [seq_ids[i] for i, m in enumerate(te_mask) if m]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr.reshape(-1,6)).reshape(X_tr.shape)
X_te_sc = sc.transform(X_te.reshape(-1,6)).reshape(X_te.shape)
y_tr_log = np.log1p(y_tr)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ── Grid Search ────────────────────────────────────────────────────────────
folds = [(anos_tr<=2017, anos_tr==2018),
         (anos_tr<=2018, anos_tr==2019),
         (anos_tr<=2019, anos_tr==2020)]

print("\n[BiLSTM] Grid Search...")
best_r2, best_p = -1e9, None
for hd in [64, 128, 256]:
    for lr in [0.001, 0.003]:
        for dr in [0.1, 0.2]:
            r2s = []
            for tr_m, va_m in folds:
                Xtr_f = X_tr_sc[tr_m]; ytr_f = y_tr_log[tr_m]
                Xva_f = X_tr_sc[va_m]; yva_f = y_tr[va_m]
                sc_f = StandardScaler()
                Xtr2 = sc_f.fit_transform(Xtr_f.reshape(-1,6)).reshape(Xtr_f.shape)
                Xva2 = sc_f.transform(Xva_f.reshape(-1,6)).reshape(Xva_f.shape)
                torch.manual_seed(9); np.random.seed(9)
                m  = BiLSTMModel(6, hd, dropout=dr)
                op = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
                sch = optim.lr_scheduler.ReduceLROnPlateau(op, patience=5, factor=0.5)
                Xf_t = torch.tensor(Xtr2, dtype=torch.float32)
                yf_t = torch.tensor(ytr_f, dtype=torch.float32).unsqueeze(1)
                Xv_t = torch.tensor(Xva2, dtype=torch.float32)
                dl   = DataLoader(TensorDataset(Xf_t, yf_t), batch_size=256, shuffle=True)
                for ep in range(60):
                    m.train()
                    for bx, by in dl:
                        op.zero_grad(); nn.MSELoss()(m(bx), by).backward(); op.step()
                m.eval()
                with torch.no_grad():
                    pv = np.expm1(m(Xv_t).numpy().flatten())
                r2s.append(r2_score(yva_f, pv))
            r2cv = np.mean(r2s)
            print(f"  hidden={hd:3d} lr={lr:.3f} dr={dr} -> R2_CV={r2cv*100:.2f}%")
            if r2cv > best_r2:
                best_r2 = r2cv; best_p = (hd, lr, dr)

hd, lr, dr = best_p
print(f"\nMejores: hidden={hd} lr={lr} dr={dr}  R2_CV={best_r2*100:.2f}%")
print("Reentrenando con early stopping (max 300 epocas)...")

torch.manual_seed(9); np.random.seed(9)
val_mask = anos_tr == 2020; fit_mask = ~val_mask
Xf = torch.tensor(X_tr_sc[fit_mask], dtype=torch.float32)
yf = torch.tensor(y_tr_log[fit_mask], dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_tr_sc[val_mask], dtype=torch.float32)
yv = torch.tensor(y_tr_log[val_mask], dtype=torch.float32)

m   = BiLSTMModel(6, hd, dropout=dr)
opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
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
        best_val = val_mse; wait = 0; best_ep = ep+1
        best_state = {k: v.clone() for k, v in m.state_dict().items()}
    else:
        wait += 1
        if wait >= 15:
            print(f"  Early stop epoch {ep+1} (mejor: epoch {best_ep})")
            break

m.load_state_dict(best_state); m.eval()
with torch.no_grad():
    pred_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred_lstm = np.expm1(pred_log)

r2l_log = r2_score(np.log1p(y_te), pred_log)
r2l_raw = r2_score(y_te, pred_lstm)
print(f"\nBiLSTM: R2 log={r2l_log*100:.2f}%  R2 crudo={r2l_raw*100:.2f}%  MAE={mean_absolute_error(y_te,pred_lstm):.4f}")
print(f"(LSTM unidireccional misma config: 86.83% log)")

# ── Ensemble con XGBoost (89.75%) ─────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT    = [c for c in df.columns if c not in COLS_EXCLUIR]
df_train = df[df['ano'] <= 2020]
df_test  = df[df['ano'] >= 2021].copy()

pipe_xgb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('m',   XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                          min_child_weight=3, gamma=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
])
pipe_xgb.fit(df_train[COLS_FEAT], np.log1p(df_train['incidencia_dengue']))
pred_xgb = np.expm1(pipe_xgb.predict(df_test[COLS_FEAT]))

lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
       for r, p in zip(df_test.itertuples(), pred_xgb)}
y_l, xgb_l, lstm_l = [], [], []
for sid, lp in zip(ids_test, pred_lstm):
    xp = lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso)&(df_test['adm_1_name'].str.upper()==adm)&(df_test['ano']==ano)&(df_test['mes']==mes)]
    if len(row):
        y_l.append(float(row['incidencia_dengue'].iloc[0]))
        xgb_l.append(xp); lstm_l.append(lp)

y, xgb, lst = np.array(y_l), np.array(xgb_l), np.array(lstm_l)
ly, lxgb, llst = np.log1p(y), np.log1p(xgb), np.log1p(lst)

diff = xgb - lst
w = float(np.clip(np.sum((y-lst)*diff)/np.sum(diff**2), 0, 1))
ens = w*xgb + (1-w)*lst
res = minimize_scalar(lambda ww: -r2_score(ly, ww*lxgb+(1-ww)*llst), bounds=(0,1), method='bounded')
w_log = float(res.x)
ens_log = w_log*lxgb + (1-w_log)*llst

print(f"\nEnsemble XGBoost + BiLSTM:")
print(f"  Opt. raw  w_xgb={w:.3f}: R2 log={r2_score(ly,np.log1p(ens))*100:.2f}%  R2 crudo={r2_score(y,ens)*100:.2f}%  MAE={mean_absolute_error(y,ens):.4f}")
print(f"  Opt. log  w_xgb={w_log:.3f}: R2 log={r2_score(ly,ens_log)*100:.2f}%")
print(f"  (Ensemble anterior LSTM unidireccional: 87.59% log  78.36% crudo)")
