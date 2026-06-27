import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
from agente_4_prediccion_dl import DengueLSTMModel, _build_sequences

SEQ_LEN = 24  # en vez de 12

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)

# Quitar 5 deptos con pico > 1000
stats = df.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok = stats[stats <= 1000].index
df = df[df.set_index(['pais','adm_1_name']).index.isin(deptos_ok)].reset_index(drop=True)

LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

print(f"Sin NIC + sin outliers: {len(df)} filas | {df['pais'].nunique()} paises")
print(f"Secuencias de {SEQ_LEN} pasos (antes: 12)")

X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, SEQ_LEN)
tr_mask = anos_seq <= 2020
te_mask = anos_seq >= 2021

X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
X_te, y_te = X_seq[te_mask], y_seq[te_mask]
anos_tr    = anos_seq[tr_mask]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr.reshape(-1, 6)).reshape(X_tr.shape)
X_te_sc = sc.transform(X_te.reshape(-1, 6)).reshape(X_te.shape)
y_tr_log = np.log1p(y_tr)

print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# Grid search con secuencias de 24
folds = [(anos_tr<=2017, anos_tr==2018),
         (anos_tr<=2018, anos_tr==2019),
         (anos_tr<=2019, anos_tr==2020)]

def quick_train(Xtr, ytr, Xva, hd, lr, dr, epochs=60, seed=9):
    torch.manual_seed(seed); np.random.seed(seed)
    m   = DengueLSTMModel(6, hd, dropout=dr)
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    Xf  = torch.tensor(Xtr, dtype=torch.float32)
    yf  = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xv  = torch.tensor(Xva, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
    for ep in range(epochs):
        m.train()
        for bx, by in dl:
            opt.zero_grad(); nn.MSELoss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xv).numpy().flatten()
        sch.step(-r2_score(np.expm1(pv[:10]), np.expm1(pv[:10])))  # dummy step
    m.eval()
    with torch.no_grad():
        pv = np.expm1(m(Xv).numpy().flatten())
    return m, pv

print("\nGrid Search (24-step sequences)...")
best_r2, best_p = -1e9, None
for hd in [128, 256, 512]:
    for lr in [0.001, 0.003]:
        for dr in [0.1, 0.2]:
            r2s = []
            for tr_m, va_m in folds:
                Xtr_f = X_tr_sc[tr_m]; ytr_f = y_tr_log[tr_m]
                Xva_f = X_tr_sc[va_m]; yva_f = y_tr[va_m]
                sc_f = StandardScaler()
                Xtr2 = sc_f.fit_transform(Xtr_f.reshape(-1,6)).reshape(Xtr_f.shape)
                Xva2 = sc_f.transform(Xva_f.reshape(-1,6)).reshape(Xva_f.shape)
                _, pv = quick_train(Xtr2, ytr_f, Xva2, hd, lr, dr)
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

m   = DengueLSTMModel(6, hd, dropout=dr)
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
pred = np.expm1(pred_log)

r2_log = r2_score(np.log1p(y_te), pred_log)
r2_raw = r2_score(y_te, pred)
mae    = mean_absolute_error(y_te, pred)

print(f"\nLSTM (seq=24, ES):")
print(f"  R2 log  : {r2_log*100:.2f}%")
print(f"  R2 crudo: {r2_raw*100:.2f}%")
print(f"  MAE     : {mae:.4f}")
print(f"  (antes seq=12: 86.83% log con mismos filtros)")
