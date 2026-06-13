# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning (LSTM PyTorch)
--------------------------------------------------
Responsabilidad: Entrenar el modelo LSTM sobre el dataset de features generado
por el Agente 2, serializar artefactos y subirlos a S3. También genera el
metrics.json combinado (LightGBM + LSTM) para el endpoint /api/metrics.
En modo inferencia, carga el modelo serializado para predicción online.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

LSTM_FEATURES = ['tmax_promedio', 'tmin_promedio', 'precipitacion',
                 'humedad_promedio', 'agua_basica', 'incidencia_dengue']
LSTM_SEQ_LEN  = 12


class DengueLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_sequences(df_sorted, features, seq_len):
    """Construye tensores de secuencias y años por departamento."""
    X_seqs, y_vals, anos = [], [], []
    for _, grp in df_sorted.groupby(['iso_a0', 'adm_1_name']):
        grp = grp.sort_values(['ano', 'mes'])
        if len(grp) < seq_len + 1:
            continue
        feat = grp[features].values.astype(np.float32)
        tgt  = grp['incidencia_dengue'].values.astype(np.float32)
        yr   = grp['ano'].values
        for i in range(seq_len, len(grp)):
            X_seqs.append(feat[i - seq_len:i])
            y_vals.append(tgt[i])
            anos.append(yr[i])
    return np.array(X_seqs), np.array(y_vals), np.array(anos)


class AgentePrediccionDL:
    def __init__(self, base_dir=None):
        if base_dir is None:
            if os.path.exists("/app"):
                self.base_dir = "/app"
            else:
                self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.db_dir    = os.path.join(self.base_dir, "Base de Datos")
        self.model_dir = os.path.join(self.db_dir, "modelos")
        self.feat_path = os.path.join(self.db_dir, "datos_procesados", "dataset_features_latam.csv")
        self.semilla   = 9

    # ─────────────────────────────────────────────────────────────
    # MODO ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────

    def entrenar_modelo(self, metricas_ml=None):
        """
        Descarga dataset_features_latam.csv de S3, entrena LSTM PyTorch con
        transformación log1p, serializa artefactos y los sube a S3.
        Genera el metrics.json combinado (LightGBM + LSTM).

        Args:
            metricas_ml: dict con r2_lgbm, mae_lgbm, n_train, n_test del Agente 3.
        """
        print("=" * 70)
        print("  ENTRENANDO — AGENTE 4: LSTM PyTorch")
        print("=" * 70)

        os.makedirs(self.model_dir, exist_ok=True)

        s3.ensure_local(s3.PREFIX_PROCESADOS + "dataset_features_latam.csv", self.feat_path)
        if not os.path.exists(self.feat_path):
            raise FileNotFoundError(f"No se encontró dataset de features: {self.feat_path}")

        df = pd.read_csv(self.feat_path)

        # Filtrado dinámico
        yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly > 100].reset_index(drop=True)

        # Construir secuencias usando solo LSTM_FEATURES (6 variables)
        X_seq, y_seq, anos_seq = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)

        train_mask = anos_seq <= 2020
        test_mask  = anos_seq >= 2021

        X_train = X_seq[train_mask]
        y_train = y_seq[train_mask]
        X_test  = X_seq[test_mask]
        y_test  = y_seq[test_mask]

        print(f"   [LSTM] Secuencias — Train: {len(X_train)} | Test: {len(X_test)}")

        # Escalar sobre datos aplanados (n_samples × seq_len, n_features)
        escalador = StandardScaler()
        X_train_flat = X_train.reshape(-1, len(LSTM_FEATURES))
        escalador.fit(X_train_flat)

        X_train_sc = escalador.transform(X_train_flat).reshape(X_train.shape)
        X_test_sc  = escalador.transform(X_test.reshape(-1, len(LSTM_FEATURES))).reshape(X_test.shape)

        # Transformación logarítmica del target
        y_train_log = np.log1p(y_train)

        X_t = torch.tensor(X_train_sc, dtype=torch.float32)
        y_t = torch.tensor(y_train_log, dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(self.semilla)
        np.random.seed(self.semilla)

        modelo = DengueLSTMModel(input_dim=len(LSTM_FEATURES))
        optimizer = optim.Adam(modelo.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.MSELoss()
        loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

        print("   [LSTM] Entrenando 80 épocas...")
        modelo.train()
        for epoch in range(80):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(modelo(bx), by)
                loss.backward()
                optimizer.step()

        # Evaluación (modelo predice en escala log → expm1 → escala real)
        modelo.eval()
        with torch.no_grad():
            pred_log = modelo(torch.tensor(X_test_sc, dtype=torch.float32)).numpy().flatten()
        pred = np.expm1(pred_log)

        r2  = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"   [LSTM] R²={r2*100:.2f}%  MAE={mae:.4f}")

        # Serializar artefactos localmente
        torch.save(modelo.state_dict(), os.path.join(self.model_dir, "lstm_model.pth"))
        with open(os.path.join(self.model_dir, "escalador_lstm.pkl"), "wb") as f:
            pickle.dump(escalador, f)
        with open(os.path.join(self.model_dir, "lstm_features.pkl"), "wb") as f:
            pickle.dump(LSTM_FEATURES, f)

        lstm_config = {"seq_len": LSTM_SEQ_LEN, "input_dim": len(LSTM_FEATURES),
                       "r2": round(r2, 4), "mae": round(mae, 4)}
        with open(os.path.join(self.model_dir, "lstm_config.json"), "w") as f:
            json.dump(lstm_config, f, indent=4)

        # Metrics combinadas (LightGBM + LSTM)
        r2_ml  = metricas_ml.get("r2_lgbm",  0.0) if metricas_ml else 0.0
        mae_ml = metricas_ml.get("mae_lgbm", 0.0) if metricas_ml else 0.0
        n_rec  = metricas_ml.get("n_train", len(df)) if metricas_ml else len(df)

        metrics = {
            "records_procesados": int(n_rec),
            "r2_lgbm":     round(r2_ml, 4),
            "mae_lgbm":    round(mae_ml, 4),
            "r2_lstm":     round(r2, 4),
            "mae_lstm":    round(mae, 4),
            "r2_ensemble": round((r2_ml + r2) / 2, 4),
            "mae_ensemble": round((mae_ml + mae) / 2, 4),
        }
        with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Subir todo a S3
        for fname in ["lstm_model.pth", "escalador_lstm.pkl", "lstm_features.pkl",
                      "lstm_config.json", "metrics.json"]:
            s3.upload(os.path.join(self.model_dir, fname), s3.PREFIX_MODELOS + fname)

        print("SUCCESS: [Agente 4] LSTM entrenado y subido a S3.")
        print("=" * 70)

        return {"r2_lstm": round(r2, 4), "mae_lstm": round(mae, 4)}

    # ─────────────────────────────────────────────────────────────
    # MODO INFERENCIA
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def cargar_modelo(cls, model_dir, base_dir=None):
        agente = cls(base_dir=base_dir)
        with open(os.path.join(model_dir, "lstm_config.json"), "r") as f:
            config = json.load(f)
        with open(os.path.join(model_dir, "lstm_features.pkl"), "rb") as f:
            agente.lstm_features = pickle.load(f)
        with open(os.path.join(model_dir, "escalador_lstm.pkl"), "rb") as f:
            agente.escalador_lstm = pickle.load(f)

        agente.lstm_seq_len = config.get("seq_len", LSTM_SEQ_LEN)
        agente.modelo_lstm  = DengueLSTMModel(
            input_dim=config.get("input_dim", len(agente.lstm_features))
        )
        agente.modelo_lstm.load_state_dict(
            torch.load(os.path.join(model_dir, "lstm_model.pth"),
                       map_location=torch.device('cpu'))
        )
        agente.modelo_lstm.eval()
        print(f"   [Agente 4] LSTM cargado — seq_len={agente.lstm_seq_len}, "
              f"input_dim={len(agente.lstm_features)}.")
        return agente

    def predecir_secuencia(self, df_dept, ref_idx, clima_overrides=None):
        start    = max(0, ref_idx - self.lstm_seq_len + 1)
        hist     = df_dept.iloc[start:ref_idx + 1]
        feat_arr = hist[self.lstm_features].values.copy().astype(float)

        if len(feat_arr) < self.lstm_seq_len:
            pad = np.zeros((self.lstm_seq_len - len(feat_arr), len(self.lstm_features)))
            feat_arr = np.vstack([pad, feat_arr])

        if clima_overrides:
            for i, fname in enumerate(self.lstm_features):
                if fname in clima_overrides:
                    feat_arr[-1, i] = float(clima_overrides[fname])
            if 'incidencia_lag1' in clima_overrides and 'incidencia_dengue' in self.lstm_features:
                feat_arr[-1, self.lstm_features.index('incidencia_dengue')] = \
                    float(clima_overrides['incidencia_lag1'])

        flat   = feat_arr.reshape(-1, len(self.lstm_features))
        scaled = self.escalador_lstm.transform(flat).reshape(1, self.lstm_seq_len, len(self.lstm_features))
        with torch.no_grad():
            pred_log = float(self.modelo_lstm(torch.tensor(scaled, dtype=torch.float32)).numpy()[0][0])
        return max(0.0, np.expm1(pred_log))
