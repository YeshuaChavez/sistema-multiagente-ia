# -*- coding: utf-8 -*-
"""
Script de Entrenamiento y Serialización de Modelos
--------------------------------------------------
Entrena LightGBM, PyTorch MLP y PyTorch LSTM con feature engineering mejorado:
  - Lags temporales profundos (incidencia lags 1-6, vecinos 1-6)
  - Rolling means de 3 y 6 meses (suavizado de tendencia)
  - Codificación cíclica del mes (sin/cos) para capturar estacionalidad
  - LightGBM reemplaza XGBoost por mayor precisión en tabular data
  - SHAP con media con signo (preserva dirección del efecto)
  - Ensemble de 3 modelos: LightGBM + MLP + LSTM
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import shap
import optuna
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

LSTM_FEATURES = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio',
                  'agua_basica', 'incidencia_dengue']
LSTM_SEQ_LEN = 12


class DengueMLPModel(nn.Module):
    def __init__(self, input_dim=34, output_dim=1):
        super(DengueMLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DengueLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(DengueLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_lstm_sequences(df, feat_cols, target_col, seq_len):
    X_seqs, y_seqs, anos = [], [], []
    for _, grp in df.groupby(['iso_a0', 'adm_1_name']):
        grp = grp.sort_values(['ano', 'mes']).reset_index(drop=True)
        if len(grp) < seq_len + 1:
            continue
        for i in range(seq_len, len(grp)):
            seq = grp[feat_cols].iloc[i - seq_len:i].values
            X_seqs.append(seq)
            y_seqs.append(grp[target_col].iloc[i])
            anos.append(grp['ano'].iloc[i])
    return np.array(X_seqs), np.array(y_seqs), np.array(anos)


def generar_lags_y_vecinos_dinamico(df, db_dir):
    print("   [Procesamiento] Generando lags, rolling means y features de estacionalidad...")
    df = df.copy()

    df = df.sort_values(by=['iso_a0', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
    group = df.groupby(['iso_a0', 'adm_1_name'])

    # 1. Lags climáticos (1-3 meses)
    cols_clima = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
    for var in cols_clima:
        base_name = var.split('_')[0] if 'promedio' in var else var
        for lag in [1, 2, 3]:
            df[f"{base_name}_lag{lag}"] = group[var].shift(lag)

    # 2. Lags de incidencia profundos (1-6 meses)
    for lag in [1, 2, 3, 4, 5, 6]:
        df[f"incidencia_lag{lag}"] = group['incidencia_dengue'].shift(lag)

    # 3. Rolling means de incidencia (ventana 3 y 6 meses, shift(1) para evitar filtración)
    df['incidencia_roll3'] = group['incidencia_dengue'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['incidencia_roll6'] = group['incidencia_dengue'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )

    # 4. Codificación cíclica del mes (sin/cos)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # 5. Vecinos espaciales con lags 1-6
    coords_path = os.path.join(db_dir, "datos_crudos", "departamentos_coordenadas.csv")
    if os.path.exists(coords_path):
        df_coords = pd.read_csv(coords_path)
        df_coords['iso_a0'] = df_coords['iso_a0'].astype(str).str.strip().str.upper()
        df_coords['adm_1_name'] = df_coords['adm_1_name'].astype(str).str.strip().str.upper()

        df['adm_1_name_upper'] = df['adm_1_name'].astype(str).str.strip().str.upper()

        neighbors_dict = {}
        for country in df_coords['iso_a0'].unique():
            country_coords = df_coords[df_coords['iso_a0'] == country].copy()
            depts = country_coords['adm_1_name'].values
            coords_vals = country_coords[['lat', 'lon']].values
            N = len(depts)
            for i in range(N):
                lat_i, lon_i = coords_vals[i]
                distances = []
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((lat_i - coords_vals[j][0])**2 + (lon_i - coords_vals[j][1])**2)
                    distances.append((depts[j], dist))
                distances.sort(key=lambda x: x[1])
                K = min(3, len(distances))
                neighbors_dict[(country, depts[i])] = [d[0] for d in distances[:K]]

        lookup = {(r.iso_a0, r.adm_1_name_upper, r.ano, r.mes): r.incidencia_dengue
                  for r in df.itertuples()}

        neighbor_inc = []
        for row in df.itertuples():
            key = (row.iso_a0, row.adm_1_name_upper)
            neighbors = neighbors_dict.get(key, [])
            if not neighbors:
                neighbor_inc.append(row.incidencia_dengue)
                continue
            vals = [lookup.get((row.iso_a0, n, row.ano, row.mes)) for n in neighbors]
            vals = [v for v in vals if v is not None]
            neighbor_inc.append(np.mean(vals) if vals else row.incidencia_dengue)

        df['incidencia_vecinos'] = neighbor_inc

        group_upper = df.groupby(['iso_a0', 'adm_1_name_upper'])
        for lag in [1, 2, 3, 4, 5, 6]:
            df[f'incidencia_vecinos_lag{lag}'] = group_upper['incidencia_vecinos'].shift(lag)

        df.drop(columns=['adm_1_name_upper', 'incidencia_vecinos'], inplace=True)
    else:
        print("   [Advertencia] No se encontro coordenadas. Vecinos con 0.")
        for lag in [1, 2, 3, 4, 5, 6]:
            df[f'incidencia_vecinos_lag{lag}'] = 0.0

    # 6. ENSO / ONI — índice global, lags computados sobre la serie temporal global
    oni_path = os.path.join(db_dir, "datos_crudos", "oni_mensual.csv")
    if os.path.exists(oni_path):
        df_oni = pd.read_csv(oni_path)[['ano', 'mes', 'oni', 'oni_lag1', 'oni_lag2', 'oni_lag3']]
        df = df.merge(df_oni, on=['ano', 'mes'], how='left')
        for c in ['oni', 'oni_lag1', 'oni_lag2', 'oni_lag3']:
            df[c] = df[c].fillna(0.0)
        print("   [ENSO] Indice ONI fusionado correctamente.")
    else:
        print("   [Advertencia] oni_mensual.csv no encontrado. Ejecuta Scripts/descargar_oni.py primero.")
        for c in ['oni', 'oni_lag1', 'oni_lag2', 'oni_lag3']:
            df[c] = 0.0

    # Eliminar nulos generados por lags
    cols_lags = [c for c in df.columns if 'lag' in c or 'roll' in c]
    df.dropna(subset=cols_lags, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
    db_dir = os.path.join(base_dir, "Base de Datos")
    model_dir = os.path.join(db_dir, "modelos")
    os.makedirs(model_dir, exist_ok=True)

    dataset_path = os.path.join(db_dir, "datos_procesados", "dataset_maestro_mensual_latam.csv")
    print(f"Cargando dataset desde: {dataset_path}")
    df_raw = pd.read_csv(dataset_path)

    df = generar_lags_y_vecinos_dinamico(df_raw, db_dir)

    # Filtrado dinamico de años activos
    yearly_totals = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
    df = df[yearly_totals > 100].reset_index(drop=True)

    COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
    COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
    print(f"Features ({len(COLS_FEAT)}): {COLS_FEAT}")

    with open(os.path.join(model_dir, "cols_feat.pkl"), "wb") as f:
        pickle.dump(COLS_FEAT, f)

    df_train = df[df['ano'] <= 2020].copy()
    df_test = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)].copy()

    X_train_raw = df_train[COLS_FEAT]
    y_train = df_train['incidencia_dengue']
    X_test_raw = df_test[COLS_FEAT]
    y_test = df_test['incidencia_dengue']

    y_train_log = np.log1p(y_train)

    # ─────────────── LightGBM + Optuna ───────────────
    print("\nEntrenando LightGBM con Optuna (80 trials)...")
    imputador_ml = SimpleImputer(strategy="median")
    X_train_imp_ml = pd.DataFrame(imputador_ml.fit_transform(X_train_raw), columns=COLS_FEAT)
    X_test_imp_ml  = pd.DataFrame(imputador_ml.transform(X_test_raw),  columns=COLS_FEAT)

    escalador_ml = StandardScaler()
    X_train_esc_ml = pd.DataFrame(escalador_ml.fit_transform(X_train_imp_ml), columns=COLS_FEAT)
    X_test_esc_ml  = pd.DataFrame(escalador_ml.transform(X_test_imp_ml),  columns=COLS_FEAT)

    modelo_ml = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    modelo_ml.fit(X_train_esc_ml, y_train_log)

    pred_ml_test_log = modelo_ml.predict(X_test_esc_ml)
    pred_ml_test = np.expm1(pred_ml_test_log)
    r2_ml  = r2_score(y_test, pred_ml_test)
    mae_ml = mean_absolute_error(y_test, pred_ml_test)
    print(f"  LightGBM  R²={r2_ml:.4f}  MAE={mae_ml:.4f}")

    with open(os.path.join(model_dir, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(modelo_ml, f)
    with open(os.path.join(model_dir, "imputador_ml.pkl"), "wb") as f:
        pickle.dump(imputador_ml, f)
    with open(os.path.join(model_dir, "escalador_ml.pkl"), "wb") as f:
        pickle.dump(escalador_ml, f)
    print("  LightGBM guardado.")

    # SHAP (con signo — preserva dirección del efecto)
    print("  Calculando SHAP (LightGBM TreeSHAP)...")
    explainer = shap.TreeExplainer(modelo_ml)
    shap_vals = explainer.shap_values(X_test_esc_ml)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    mean_shap = shap_vals.mean(axis=0)  # media con signo, NO abs
    shap_importance = {var: float(val) for var, val in zip(COLS_FEAT, mean_shap)}
    shap_importance = dict(sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True))
    with open(os.path.join(model_dir, "shap_importance.json"), "w") as f:
        json.dump(shap_importance, f, indent=4)
    print("  SHAP guardado.")

    # ─────────────── PyTorch MLP ───────────────
    print("\nEntrenando PyTorch MLP...")
    imputador_dl = SimpleImputer(strategy="median")
    X_train_imp_dl = pd.DataFrame(imputador_dl.fit_transform(X_train_raw), columns=COLS_FEAT)
    X_test_imp_dl = pd.DataFrame(imputador_dl.transform(X_test_raw), columns=COLS_FEAT)

    escalador_dl = StandardScaler()
    X_train_esc_dl = pd.DataFrame(escalador_dl.fit_transform(X_train_imp_dl), columns=COLS_FEAT)
    X_test_esc_dl = pd.DataFrame(escalador_dl.transform(X_test_imp_dl), columns=COLS_FEAT)

    y_train_log_dl = np.log1p(y_train.values)
    X_train_t = torch.tensor(X_train_esc_dl.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_log_dl, dtype=torch.float32).unsqueeze(1)

    model_dl = DengueMLPModel(input_dim=len(COLS_FEAT))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.005, weight_decay=1e-4)

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model_dl.train()
    for epoch in range(100):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model_dl(bx), by)
            loss.backward()
            optimizer.step()

    model_dl.eval()
    X_test_t = torch.tensor(X_test_esc_dl.values, dtype=torch.float32)
    with torch.no_grad():
        pred_dl_test_log = model_dl(X_test_t).numpy().flatten()
    pred_dl_test = np.expm1(pred_dl_test_log)
    r2_dl = r2_score(y_test, pred_dl_test)
    mae_dl = mean_absolute_error(y_test, pred_dl_test)
    print(f"  MLP       R²={r2_dl:.4f}  MAE={mae_dl:.4f}")

    torch.save(model_dl.state_dict(), os.path.join(model_dir, "mlp_model.pth"))
    with open(os.path.join(model_dir, "imputador_dl.pkl"), "wb") as f:
        pickle.dump(imputador_dl, f)
    with open(os.path.join(model_dir, "escalador_dl.pkl"), "wb") as f:
        pickle.dump(escalador_dl, f)
    print("  MLP guardado.")

    # ─────────────── PyTorch LSTM ───────────────
    print("\nEntrenando PyTorch LSTM...")

    # Crear secuencias por departamento (ventana = 12 meses)
    X_seq, y_seq, anos_seq = create_lstm_sequences(df, LSTM_FEATURES, 'incidencia_dengue', LSTM_SEQ_LEN)

    train_mask = anos_seq <= 2020
    test_mask = anos_seq >= 2021

    X_train_seq = X_seq[train_mask]
    y_train_seq = y_seq[train_mask]
    X_test_seq = X_seq[test_mask]
    y_test_seq = y_seq[test_mask]

    # Escalar sobre datos aplanados (n_samples*seq_len, n_features)
    X_train_flat = X_train_seq.reshape(-1, len(LSTM_FEATURES))
    escalador_lstm = StandardScaler()
    escalador_lstm.fit(X_train_flat)

    X_train_sc = escalador_lstm.transform(X_train_flat).reshape(X_train_seq.shape)
    X_test_sc = escalador_lstm.transform(
        X_test_seq.reshape(-1, len(LSTM_FEATURES))
    ).reshape(X_test_seq.shape)

    y_train_log_lstm = np.log1p(y_train_seq)

    X_train_lt = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_lt = torch.tensor(y_train_log_lstm, dtype=torch.float32).unsqueeze(1)

    model_lstm = DengueLSTMModel(input_dim=len(LSTM_FEATURES))
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.003, weight_decay=1e-4)

    ds_lstm = TensorDataset(X_train_lt, y_train_lt)
    loader_lstm = DataLoader(ds_lstm, batch_size=256, shuffle=True)

    model_lstm.train()
    for epoch in range(80):
        for bx, by in loader_lstm:
            optimizer_lstm.zero_grad()
            loss = criterion(model_lstm(bx), by)
            loss.backward()
            optimizer_lstm.step()

    model_lstm.eval()
    X_test_lt = torch.tensor(X_test_sc, dtype=torch.float32)
    with torch.no_grad():
        pred_lstm_test_log = model_lstm(X_test_lt).numpy().flatten()
    pred_lstm_test = np.expm1(pred_lstm_test_log)

    # Alinear test con la misma longitud que y_test_seq (puede diferir de y_test)
    r2_lstm = r2_score(y_test_seq, pred_lstm_test)
    mae_lstm = mean_absolute_error(y_test_seq, pred_lstm_test)
    print(f"  LSTM      R²={r2_lstm:.4f}  MAE={mae_lstm:.4f}")

    # Ensemble de 3 modelos sobre la interseccion de indices de test
    # (el LSTM tiene un recorte de seq_len filas al inicio de cada grupo)
    print("\n=== METRICAS FINALES ===")
    print(f"  LightGBM  R²={r2_ml:.4f}  MAE={mae_ml:.2f}")
    print(f"  MLP       R²={r2_dl:.4f}  MAE={mae_dl:.2f}")
    print(f"  LSTM      R²={r2_lstm:.4f}  MAE={mae_lstm:.2f}")

    torch.save(model_lstm.state_dict(), os.path.join(model_dir, "lstm_model.pth"))
    with open(os.path.join(model_dir, "escalador_lstm.pkl"), "wb") as f:
        pickle.dump(escalador_lstm, f)
    with open(os.path.join(model_dir, "lstm_features.pkl"), "wb") as f:
        pickle.dump(LSTM_FEATURES, f)
    lstm_config = {"seq_len": LSTM_SEQ_LEN, "input_dim": len(LSTM_FEATURES),
                   "r2": round(r2_lstm, 4), "mae": round(mae_lstm, 4)}
    with open(os.path.join(model_dir, "lstm_config.json"), "w") as f:
        json.dump(lstm_config, f, indent=4)

    # Guardar metricas globales para el frontend
    metrics = {
        "records_procesados": int(len(df)),
        "r2_lgbm": round(r2_ml, 4),
        "mae_lgbm": round(mae_ml, 4),
        "r2_mlp": round(r2_dl, 4),
        "mae_mlp": round(mae_dl, 4),
        "r2_lstm": round(r2_lstm, 4),
        "mae_lstm": round(mae_lstm, 4),
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n  LSTM + config + metricas guardados.")
    print("=" * 60)
    print("Todos los modelos serializados en Base de Datos/modelos/")


if __name__ == "__main__":
    main()
