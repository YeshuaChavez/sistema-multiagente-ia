import sys, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

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

LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)
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

# Mejores params del grid: hidden=512, lr=0.003, dropout=0.2
hidden_dim, lr, dropout = 512, 0.003, 0.2
print(f"Entrenando: hidden={hidden_dim} lr={lr} dropout={dropout}")
print(f"Early stopping patience=15, max 300 epocas, ReduceLROnPlateau")

torch.manual_seed(9); np.random.seed(9)

val_mask = anos_tr == 2020
fit_mask = ~val_mask

Xf = torch.tensor(X_tr_sc[fit_mask], dtype=torch.float32)
yf = torch.tensor(y_tr_log[fit_mask], dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_tr_sc[val_mask], dtype=torch.float32)
yv_log = torch.tensor(y_tr_log[val_mask], dtype=torch.float32)

m   = DengueLSTMModel(6, hidden_dim, dropout=dropout)
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
        pv_log = m(Xv).flatten()
        val_mse = float(nn.MSELoss()(pv_log, yv_log))
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
    pred_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred = np.expm1(pred_log)

r2_log = r2_score(np.log1p(y_te), pred_log)
r2_raw = r2_score(y_te, pred)
mae    = mean_absolute_error(y_te, pred)

print(f"\nLSTM (ES + LR Scheduler):")
print(f"  R2 log  : {r2_log*100:.2f}%")
print(f"  R2 crudo: {r2_raw*100:.2f}%")
print(f"  MAE     : {mae:.4f}")
print(f"  (antes sin NIC: 83.05% log)")
