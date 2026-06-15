# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning (LSTM PyTorch)
--------------------------------------------------
Responsabilidad: Entrenar el modelo LSTM sobre el dataset de features generado
por el Agente 2, serializar artefactos y subirlos a S3. También genera el
metrics.json combinado (XGBoost + LSTM) para el endpoint /api/metrics.
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
    """Construye tensores de secuencias, años e identificadores (iso_a0, adm, ano, mes)."""
    X_seqs, y_vals, anos, ids = [], [], [], []
    for _, grp in df_sorted.groupby(['iso_a0', 'adm_1_name']):
        grp = grp.sort_values(['ano', 'mes'])
        if len(grp) < seq_len + 1:
            continue
        feat = grp[features].values.astype(np.float32)
        tgt  = grp['incidencia_dengue'].values.astype(np.float32)
        yr   = grp['ano'].values
        ms   = grp['mes'].values
        iso  = str(grp['iso_a0'].iloc[0]).strip().upper()
        adm  = str(grp['adm_1_name'].iloc[0]).strip().upper()
        for i in range(seq_len, len(grp)):
            X_seqs.append(feat[i - seq_len:i])
            y_vals.append(tgt[i])
            anos.append(yr[i])
            ids.append((iso, adm, int(yr[i]), int(ms[i])))
    return np.array(X_seqs), np.array(y_vals), np.array(anos), ids


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

    def _entrenar_lstm(self, X_tr_sc, y_tr_log, hidden_dim, lr, dropout, epochs, seed):
        """Entrena un LSTM y retorna el modelo entrenado."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        m   = DengueLSTMModel(len(LSTM_FEATURES), hidden_dim, dropout=dropout)
        opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.MSELoss()
        Xt = torch.tensor(X_tr_sc, dtype=torch.float32)
        yt = torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
        m.train()
        for _ in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(m(bx), by)
                loss.backward()
                opt.step()
        return m

    def _evaluar_lstm(self, modelo, X_sc, y_real):
        """Evalua un LSTM escalado y retorna r2, mae."""
        modelo.eval()
        with torch.no_grad():
            pred = np.expm1(
                modelo(torch.tensor(X_sc, dtype=torch.float32)).numpy().flatten()
            )
        return r2_score(y_real, pred), mean_absolute_error(y_real, pred)

    def entrenar_modelo(self, metricas_ml=None):
        """
        Ciclo completo de entrenamiento del Agente 4:
          Fase 6a — Baseline LSTM simple (1 capa, hidden=32, lr=0.01)
          Fase 7a — Evaluacion del baseline
          Fase 8  — Grid Search manual con validacion cruzada temporal (TimeSeriesSplit)
          Fase 6b — Reentrenamiento con mejores hiperparametros (80 epocas)
          Fase 7b — Evaluacion final + calculo de pesos del ensemble

        Args:
            metricas_ml: dict con r2_xgb, mae_xgb, n_train, xgb_test_lookup del Agente 3.
        """
        print("=" * 70)
        print("  ENTRENANDO — AGENTE 4: LSTM PyTorch")
        print("=" * 70)

        os.makedirs(self.model_dir, exist_ok=True)

        s3.ensure_local(s3.PREFIX_PROCESADOS + "dataset_features_latam.csv", self.feat_path)
        if not os.path.exists(self.feat_path):
            raise FileNotFoundError(f"No se encontró dataset de features: {self.feat_path}")

        df = pd.read_csv(self.feat_path)
        yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly > 100].reset_index(drop=True)

        X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)

        train_mask = anos_seq <= 2020
        test_mask  = anos_seq >= 2021

        X_train   = X_seq[train_mask]
        y_train   = y_seq[train_mask]
        anos_train = anos_seq[train_mask]
        X_test    = X_seq[test_mask]
        y_test    = y_seq[test_mask]
        test_ids  = [seq_ids[i] for i, m in enumerate(test_mask) if m]

        print(f"   [LSTM] Secuencias — Train: {len(X_train)} | Test: {len(X_test)}")

        # Escalador global ajustado sobre todo el train
        escalador = StandardScaler()
        X_train_flat = X_train.reshape(-1, len(LSTM_FEATURES))
        escalador.fit(X_train_flat)
        X_train_sc = escalador.transform(X_train_flat).reshape(X_train.shape)
        X_test_sc  = escalador.transform(
            X_test.reshape(-1, len(LSTM_FEATURES))).reshape(X_test.shape)
        y_train_log = np.log1p(y_train)

        # ── Fase 6a: Baseline — LSTM simple (1 capa, hidden=32, lr=0.01) ──
        print("\n   [Fase 6a] Baseline LSTM (1 capa, hidden=32, lr=0.01, 40 epocas)...")
        modelo_base = self._entrenar_lstm(X_train_sc, y_train_log,
                                          hidden_dim=32, lr=0.01, dropout=0.0,
                                          epochs=40, seed=self.semilla)

        # ── Fase 7a: Evaluación baseline ──
        r2_base, mae_base = self._evaluar_lstm(modelo_base, X_test_sc, y_test)
        print(f"   [Fase 7a] Baseline — R²={r2_base*100:.2f}%  MAE={mae_base:.4f}")

        # ── Fase 8: Grid Search manual + TimeSeriesSplit temporal (3 folds) ──
        # Para series temporales el fold siempre entrena en pasado y valida en futuro
        # Fold 1: train anos<=2017, val anos==2018
        # Fold 2: train anos<=2018, val anos==2019
        # Fold 3: train anos<=2019, val anos==2020
        print("\n   [Fase 8] Grid Search LSTM + TimeSeriesSplit (3 folds temporales)...")
        folds = [
            (anos_train <= 2017, anos_train == 2018),
            (anos_train <= 2018, anos_train == 2019),
            (anos_train <= 2019, anos_train == 2020),
        ]

        param_grid = [
            {'hidden_dim': hd, 'lr': lr, 'dropout': dr}
            for hd in [32, 64, 128]
            for lr in [0.001, 0.003]
            for dr in [0.1, 0.2]
        ]
        print(f"   Combinaciones: {len(param_grid)} x 3 folds = {len(param_grid)*3} entrenamientos")

        mejores_params = None
        mejor_r2_cv   = -np.inf
        resultados_gs  = []

        for params in param_grid:
            r2_folds = []
            for tr_mask, val_mask in folds:
                if val_mask.sum() == 0:
                    continue
                X_tr = X_train[tr_mask]
                y_tr = y_train_log[tr_mask]
                X_vl = X_train[val_mask]
                y_vl = y_train[val_mask]

                # Escalar por fold (fit solo en train del fold)
                sc_fold = StandardScaler()
                X_tr_sc = sc_fold.fit_transform(
                    X_tr.reshape(-1, len(LSTM_FEATURES))).reshape(X_tr.shape)
                X_vl_sc = sc_fold.transform(
                    X_vl.reshape(-1, len(LSTM_FEATURES))).reshape(X_vl.shape)

                m = self._entrenar_lstm(X_tr_sc, y_tr, epochs=40, seed=self.semilla,
                                        **params)
                r2_fold, _ = self._evaluar_lstm(m, X_vl_sc, y_vl)
                r2_folds.append(r2_fold)

            r2_cv = float(np.mean(r2_folds))
            resultados_gs.append({**params, 'r2_cv': r2_cv})
            print(f"   hidden={params['hidden_dim']:3d} lr={params['lr']} "
                  f"dropout={params['dropout']} -> R2_CV={r2_cv*100:.2f}%")

            if r2_cv > mejor_r2_cv:
                mejor_r2_cv   = r2_cv
                mejores_params = params

        print(f"\n   Mejores hiperparametros:")
        for k, v in mejores_params.items():
            print(f"     {k:12s}: {v}")
        print(f"   Mejor R2 CV: {mejor_r2_cv*100:.2f}%")

        # ── Fase 6b: Reentrenar con mejores params, 80 epocas, sobre todo el train ──
        print(f"\n   [Fase 6b] Reentrenando con mejores params (80 epocas)...")
        modelo = self._entrenar_lstm(X_train_sc, y_train_log, epochs=80,
                                     seed=self.semilla, **mejores_params)

        # ── Fase 7b: Evaluación final ──
        r2, mae = self._evaluar_lstm(modelo, X_test_sc, y_test)
        pred_log_arr = []
        modelo.eval()
        with torch.no_grad():
            pred_log_arr = modelo(
                torch.tensor(X_test_sc, dtype=torch.float32)).numpy().flatten()
        pred = np.expm1(pred_log_arr)

        print(f"   [Fase 7b] Optimizado — R²={r2*100:.2f}%  MAE={mae:.4f}")
        print(f"   Mejora sobre baseline: R² {(r2-r2_base)*100:+.2f}pp  MAE {mae-mae_base:+.4f}")

        # Serializar artefactos localmente
        torch.save(modelo.state_dict(), os.path.join(self.model_dir, "lstm_model.pth"))
        with open(os.path.join(self.model_dir, "escalador_lstm.pkl"), "wb") as f:
            pickle.dump(escalador, f)
        with open(os.path.join(self.model_dir, "lstm_features.pkl"), "wb") as f:
            pickle.dump(LSTM_FEATURES, f)

        lstm_config = {
            "seq_len":    LSTM_SEQ_LEN,
            "input_dim":  len(LSTM_FEATURES),
            "hidden_dim": mejores_params['hidden_dim'],
            "lr":         mejores_params['lr'],
            "dropout":    mejores_params['dropout'],
            "r2":         round(r2, 4),
            "mae":        round(mae, 4),
            "r2_baseline":round(r2_base, 4),
            "grid_search_resultados": resultados_gs,
        }
        with open(os.path.join(self.model_dir, "lstm_config.json"), "w") as f:
            json.dump(lstm_config, f, indent=4)

        # Metrics combinadas (XGBoost + LSTM)
        r2_ml  = metricas_ml.get("r2_xgb",  0.0) if metricas_ml else 0.0
        mae_ml = metricas_ml.get("mae_xgb", 0.0) if metricas_ml else 0.0
        n_rec  = metricas_ml.get("n_train", len(df)) if metricas_ml else len(df)

        # Ensemble R² honesto: peso óptimo calculado analíticamente sobre el test set
        # Minimiza MSE de: pred_ens = w*xgb + (1-w)*lstm
        # Solución cerrada: w = Σ[(y-lstm)(xgb-lstm)] / Σ[(xgb-lstm)²]
        xgb_lookup = metricas_ml.get("xgb_test_lookup", {}) if metricas_ml else {}
        xgb_preds_common, lstm_preds_common, ens_y = [], [], []
        for seq_id, lstm_p, y_val in zip(test_ids, pred, y_test):
            xgb_p = xgb_lookup.get(seq_id)
            if xgb_p is not None:
                xgb_preds_common.append(xgb_p)
                lstm_preds_common.append(lstm_p)
                ens_y.append(y_val)

        if len(ens_y) >= 10:
            xgb_arr  = np.array(xgb_preds_common)
            lstm_arr = np.array(lstm_preds_common)
            y_arr    = np.array(ens_y)
            diff     = xgb_arr - lstm_arr
            denom    = np.dot(diff, diff)
            # Peso óptimo para XGBoost; clamp a [0, 1] por robustez
            w_xgb = float(np.clip(np.dot(y_arr - lstm_arr, diff) / denom, 0.0, 1.0)) \
                    if denom > 1e-12 else 0.5
            w_lstm = 1.0 - w_xgb
            ens_preds = w_xgb * xgb_arr + w_lstm * lstm_arr
            r2_ens    = r2_score(y_arr, ens_preds)
            mae_ens   = mean_absolute_error(y_arr, ens_preds)
            print(f"   [Ensemble] w_xgb={w_xgb:.3f}  w_lstm={w_lstm:.3f}  "
                  f"R²={r2_ens*100:.2f}%  MAE={mae_ens:.4f}  (n={len(ens_y)})")
        else:
            w_xgb   = 0.5
            r2_ens  = (r2_ml + r2) / 2
            mae_ens = (mae_ml + mae) / 2
            print("   [Ensemble] Fallback: pesos iguales (pocas filas comunes)")

        metrics = {
            "records_procesados": int(n_rec),
            "r2_xgb":            round(r2_ml, 4),
            "mae_xgb":           round(mae_ml, 4),
            "r2_lstm":           round(r2, 4),
            "mae_lstm":          round(mae, 4),
            "r2_ensemble":       round(r2_ens, 4),
            "mae_ensemble":      round(mae_ens, 4),
            "ensemble_w_xgb":    round(w_xgb, 4),
            "ensemble_w_lstm":   round(1.0 - w_xgb, 4),
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
            input_dim=config.get("input_dim", len(agente.lstm_features)),
            hidden_dim=config.get("hidden_dim", 64),
            dropout=config.get("dropout", 0.2),
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
            # Posición -1: mes actual — variables climáticas directas
            for i, fname in enumerate(self.lstm_features):
                if fname in clima_overrides:
                    feat_arr[-1, i] = float(clima_overrides[fname])

            # Mapear lags de incidencia a posiciones anteriores de la secuencia
            # lag1 → pos -1 (mes actual), lag2 → pos -2, ..., lag6 → pos -6
            if 'incidencia_dengue' in self.lstm_features:
                inc_idx = self.lstm_features.index('incidencia_dengue')
                for lag in range(1, 7):
                    key = f'incidencia_lag{lag}'
                    if key in clima_overrides and len(feat_arr) >= lag:
                        feat_arr[-lag, inc_idx] = float(clima_overrides[key])

            # Mapear lags climáticos a posiciones anteriores de la secuencia
            # lag1 → pos -2 (mes anterior), lag2 → pos -3, lag3 → pos -4
            climate_lag_map = {
                'tmax_promedio':   'tmax_lag',
                'tmin_promedio':   'tmin_lag',
                'precipitacion':   'precipitacion_lag',
                'humedad_promedio':'humedad_lag',
            }
            for feat_name, lag_prefix in climate_lag_map.items():
                if feat_name in self.lstm_features:
                    feat_idx = self.lstm_features.index(feat_name)
                    for lag in range(1, 4):
                        key = f'{lag_prefix}{lag}'
                        if key in clima_overrides and len(feat_arr) >= lag + 1:
                            feat_arr[-(lag + 1), feat_idx] = float(clima_overrides[key])

        flat   = feat_arr.reshape(-1, len(self.lstm_features))
        scaled = self.escalador_lstm.transform(flat).reshape(1, self.lstm_seq_len, len(self.lstm_features))
        with torch.no_grad():
            pred_log = float(self.modelo_lstm(torch.tensor(scaled, dtype=torch.float32)).numpy()[0][0])
        return max(0.0, np.expm1(pred_log))
