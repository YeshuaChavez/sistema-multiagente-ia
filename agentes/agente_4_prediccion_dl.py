# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning
--------------------------------------------------
Responsabilidad: Modelamiento temporal mediante una arquitectura LSTM (Long Short-Term
Memory) en PyTorch. Captura dependencias temporales de largo plazo usando secuencias
de 12 meses de variables climáticas y epidemiológicas.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

LSTM_FEATURES = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio',
                 'agua_basica', 'incidencia_dengue']
LSTM_SEQ_LEN = 12


class DengueLSTMModel(nn.Module):
    """LSTM apilado para regresión de incidencia de dengue."""
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_sequences(df_sorted, features, target, seq_len):
    """Construye tensores de secuencias (X) y valores objetivo (y) por departamento."""
    X_seqs, y_vals = [], []
    for _, grp in df_sorted.groupby(['iso_a0', 'adm_1_name']):
        grp = grp.sort_values(['ano', 'mes'])
        feat_arr = grp[features].values.astype(np.float32)
        tgt_arr = grp[target].values.astype(np.float32)
        for i in range(seq_len, len(grp)):
            X_seqs.append(feat_arr[i - seq_len:i])
            y_vals.append(tgt_arr[i])
    return np.array(X_seqs), np.array(y_vals)


class AgentePrediccionDL:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.dataset_path = os.path.join(self.db_dir, "datos_procesados",
                                         "dataset_maestro_mensual_latam.csv")
        self.semilla = 9

    def generar_lags_y_vecinos_dinamico(self, df):
        """Calcula dinámicamente rezagos temporales y espaciales (vecinos)."""
        print("   [Memoria] Calculando rezagos temporales y espaciales...")
        df = df.copy()

        df = df.sort_values(by=['iso_a0', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        group = df.groupby(['iso_a0', 'adm_1_name'])

        cols_clima = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
        for var in cols_clima:
            base_name = var.split('_')[0] if 'promedio' in var else var
            for lag in [1, 2, 3]:
                df[f"{base_name}_lag{lag}"] = group[var].shift(lag)

        for lag in [1, 2, 3]:
            df[f"incidencia_lag{lag}"] = group['incidencia_dengue'].shift(lag)

        coords_path = os.path.join(self.db_dir, "datos_crudos", "departamentos_coordenadas.csv")
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
                    distances = []
                    for j in range(N):
                        if i == j:
                            continue
                        dist = np.sqrt((coords_vals[i][0] - coords_vals[j][0])**2 +
                                       (coords_vals[i][1] - coords_vals[j][1])**2)
                        distances.append((depts[j], dist))
                    distances.sort(key=lambda x: x[1])
                    neighbors_dict[(country, depts[i])] = [d[0] for d in distances[:min(3, len(distances))]]

            lookup = {(r.iso_a0, r.adm_1_name_upper, r.ano, r.mes): r.incidencia_dengue
                      for r in df.itertuples()}

            neighbor_inc = []
            for row in df.itertuples():
                key = (row.iso_a0, row.adm_1_name_upper)
                neighbors = neighbors_dict.get(key, [])
                vals = [lookup.get((row.iso_a0, n, row.ano, row.mes)) for n in neighbors]
                vals = [v for v in vals if v is not None]
                neighbor_inc.append(np.mean(vals) if vals else row.incidencia_dengue)

            df['incidencia_vecinos'] = neighbor_inc
            group_upper = df.groupby(['iso_a0', 'adm_1_name_upper'])
            for lag in [1, 2, 3]:
                df[f'incidencia_vecinos_lag{lag}'] = group_upper['incidencia_vecinos'].shift(lag)
            df.drop(columns=['adm_1_name_upper', 'incidencia_vecinos'], inplace=True)
        else:
            print("   [Advertencia] No se encontró el archivo de coordenadas. Omitiendo vecinos.")
            for lag in [1, 2, 3]:
                df[f'incidencia_vecinos_lag{lag}'] = 0.0

        cols_lags = [c for c in df.columns if 'lag' in c]
        df.dropna(subset=cols_lags, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _train_lstm(self, X_seq, y_arr, epochs=80, lr=0.003, batch_size=256):
        """Entrena el modelo LSTM sobre secuencias pre-construidas."""
        torch.manual_seed(self.semilla)
        np.random.seed(self.semilla)

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        model = DengueLSTMModel(input_dim=X_seq.shape[2])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        return model

    def entrenar_modelo(self):
        """
        Carga el dataset mensual, construye secuencias temporales de 12 meses
        y entrena el modelo LSTM PyTorch (partición cronológica 2014-2020 / 2021-2022).
        """
        print("[Agente 4] Cargando dataset maestro mensual consolidado...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset maestro '{self.dataset_path}'.")

        df_raw = pd.read_csv(self.dataset_path)
        df = self.generar_lags_y_vecinos_dinamico(df_raw)

        print("   [DL/LSTM] Filtrando dataset hasta el año 2022...")
        df = df[df['ano'] <= 2022].reset_index(drop=True)

        print("   [DL/LSTM] Aplicando filtrado dinámico de años activos (>100 casos país-año)...")
        yearly_totals = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly_totals > 100].reset_index(drop=True)

        # Imputar y escalar las 6 features LSTM antes de construir secuencias
        imputador = SimpleImputer(strategy="median")
        escalador = StandardScaler()

        df_lstm = df[['iso_a0', 'adm_1_name', 'ano', 'mes'] + LSTM_FEATURES].copy()
        df_lstm[LSTM_FEATURES] = imputador.fit_transform(df_lstm[LSTM_FEATURES])
        df_lstm[LSTM_FEATURES] = escalador.fit_transform(df_lstm[LSTM_FEATURES])

        # Partición cronológica antes de construir secuencias
        df_train_raw = df[df['ano'] <= 2020]
        df_test_raw = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)]

        df_lstm_train = df_lstm[df_lstm['ano'] <= 2020]
        df_lstm_test = df_lstm[(df_lstm['ano'] >= 2021) & (df_lstm['ano'] <= 2022)]

        print("   [DL/LSTM] Construyendo secuencias de 12 meses...")
        X_train, y_train = _build_sequences(df_lstm, LSTM_FEATURES, 'incidencia_dengue', LSTM_SEQ_LEN)

        # Para el test, necesitamos las últimas SEQ_LEN filas de train de cada dept como contexto
        df_lstm_ctx = df_lstm[df_lstm['ano'] <= 2020]
        df_lstm_all = df_lstm.copy()
        X_test_seqs, y_test_vals = [], []
        for (iso, adm), grp in df_lstm_all.groupby(['iso_a0', 'adm_1_name']):
            grp = grp.sort_values(['ano', 'mes'])
            feat_arr = grp[LSTM_FEATURES].values.astype(np.float32)
            tgt_arr = df[(df['iso_a0'] == iso) & (df['adm_1_name'] == adm)].sort_values(
                ['ano', 'mes'])['incidencia_dengue'].values.astype(np.float32)
            test_mask = (grp['ano'] >= 2021).values
            for i in range(LSTM_SEQ_LEN, len(grp)):
                if test_mask[i]:
                    X_test_seqs.append(feat_arr[i - LSTM_SEQ_LEN:i])
                    if i < len(tgt_arr):
                        y_test_vals.append(tgt_arr[i])
        X_test = np.array(X_test_seqs, dtype=np.float32)
        y_test = np.array(y_test_vals, dtype=np.float32)

        # Mediana de cada LSTM feature sobre el train (escala original, para live prediction)
        lstm_feat_medians = {f: float(df[df['ano'] <= 2020][f].median()) for f in LSTM_FEATURES}

        # Entrenamiento final
        print(f"   [DL/LSTM] Entrenando LSTM — {X_train.shape[0]} secuencias, shape {X_train.shape}...")
        modelo_lstm = self._train_lstm(X_train, y_train, epochs=80, lr=0.003)

        # Evaluación sobre Test
        print("   [DL/LSTM] Evaluando LSTM sobre Test Set (2021-2022)...")
        modelo_lstm.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_pred = modelo_lstm(X_test_t).numpy().flatten()
            y_pred_raw = escalador.inverse_transform(
                np.column_stack([np.zeros((len(y_pred), len(LSTM_FEATURES) - 1)), y_pred])
            )[:, -1]
            y_pred_raw = np.clip(y_pred_raw, 0.0, None)

        y_test_real = y_test

        test_mae = mean_absolute_error(y_test_real, y_pred_raw)
        test_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_raw))
        test_r2 = r2_score(y_test_real, y_pred_raw)
        print(f"   [DL/LSTM] Test (21-22): MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | R²={test_r2*100:.2f}%")

        print("SUCCESS: [Agente 4] Entrenamiento LSTM PyTorch completado.")
        print("=" * 70)

        return {
            "modelo": modelo_lstm,
            "lstm_features": LSTM_FEATURES,
            "lstm_seq_len": LSTM_SEQ_LEN,
            "lstm_imputador": imputador,
            "lstm_scaler": escalador,
            "lstm_feat_medians": lstm_feat_medians,
            # Live prediction map: incidencia_dengue → incidencia_lag1 (proxy en sliders)
            "lstm_live_map": {"incidencia_dengue": "incidencia_lag1"},
            # Compatibility keys for agente_5 metrics display
            "imputador": imputador,
            "escalador": escalador,
            "cols_feat": LSTM_FEATURES,
            "y_test": y_test_real,
            "X_test": X_test,
            "cv_mae": test_mae,
            "cv_rmse": test_rmse,
            "cv_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "y_pred": y_pred_raw,
        }


if __name__ == "__main__":
    agente = AgentePrediccionDL()
    resultados = agente.entrenar_modelo()
