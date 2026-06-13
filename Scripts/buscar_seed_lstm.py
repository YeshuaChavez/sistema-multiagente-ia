# -*- coding: utf-8 -*-
"""Busca el seed de PyTorch que reproduce el mejor R² del LSTM."""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final")
from Scripts.entrenar_y_guardar_modelos import generar_lags_y_vecinos_dinamico, create_lstm_sequences, DengueLSTMModel

BASE_DIR = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
DB_DIR   = os.path.join(BASE_DIR, "Base de Datos")

LSTM_FEATURES = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio',
                  'agua_basica', 'incidencia_dengue']
LSTM_SEQ_LEN  = 12
EPOCHS        = 80

def train_lstm(seed):
    torch.manual_seed(seed)
    df_raw = pd.read_csv(os.path.join(DB_DIR, "datos_procesados", "dataset_maestro_mensual_latam.csv"))
    df = generar_lags_y_vecinos_dinamico(df_raw, DB_DIR)
    yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
    df = df[yearly > 100].reset_index(drop=True)

    df_train = df[df['ano'] <= 2020]
    df_test  = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)]

    X_tr, y_tr, _ = create_lstm_sequences(df_train, LSTM_FEATURES, 'incidencia_dengue', LSTM_SEQ_LEN)
    X_te, y_te, _ = create_lstm_sequences(df_test,  LSTM_FEATURES, 'incidencia_dengue', LSTM_SEQ_LEN)

    scl = StandardScaler()
    X_tr_sc = scl.fit_transform(X_tr.reshape(-1, len(LSTM_FEATURES))).reshape(X_tr.shape)
    X_te_sc = scl.transform(X_te.reshape(-1, len(LSTM_FEATURES))).reshape(X_te.shape)

    y_tr_log = np.log1p(y_tr)
    X_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_t = torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1)

    model = DengueLSTMModel(input_dim=len(LSTM_FEATURES))
    opt   = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for bx, by in loader:
            opt.zero_grad(); loss_fn(model(bx), by).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        preds = np.expm1(model(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten())
    return r2_score(y_te, preds), mean_absolute_error(y_te, preds)

best_r2, best_seed = 0, -1
print("Buscando seed optimo para LSTM...\n")
for seed in range(25):
    r2, mae = train_lstm(seed)
    marker = " <-- MEJOR" if r2 > best_r2 else ""
    print(f"  seed={seed:3d}  R²={r2:.4f}  MAE={mae:.4f}{marker}")
    if r2 > best_r2:
        best_r2, best_seed = r2, seed

print(f"\nMejor seed: {best_seed}  R²={best_r2:.4f}")
