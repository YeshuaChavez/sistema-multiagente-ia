# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning (LSTM PyTorch)
--------------------------------------------------
Responsabilidad: Entrenar el modelo LSTM sobre el dataset de features generado
por el Agente 2, evaluarlo (R², MAE, RMSE) y serializar artefactos a S3.
Retorna sus métricas y las predicciones del test set para que el Agente 5
(orquestador) las combine con las del Agente 3 y arme el ensemble.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

LSTM_SEQ_LEN  = 12
# Features determinados dinámicamente en entrenar_modelo() desde el dataset
LSTM_FEATURES = None  # placeholder — se sobreescribe en entrenamiento e inferencia


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

        self.model_dir = os.path.join(self.base_dir, "data", "models")
        self.feat_path = os.path.join(self.base_dir, "data", "processed", "dataset_features_latam.csv")
        self.semilla   = 42

    # ─────────────────────────────────────────────────────────────
    # MODO ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────

    def _entrenar_lstm(self, X_tr_sc, y_tr_log, hidden_dim, lr, dropout, epochs, seed, num_layers=2):
        """Entrena un LSTM y retorna el modelo entrenado."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        m   = DengueLSTMModel(X_tr_sc.shape[2], hidden_dim, num_layers=num_layers, dropout=dropout)
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

    def entrenar_modelo(self):
        """
        Ciclo de vida completo del modelo DL (Agente 4 — LSTM PyTorch).
        Entrena de forma independiente del Agente 3 — no recibe ni depende de
        métricas de XGBoost; la combinación de ambos modelos (ensemble +
        clasificación de riesgo) es responsabilidad exclusiva del Agente 5.

          Fase 1  — Definicion del problema: prediccion de tasa de incidencia de dengue
                    a escala subnacional mensual (ver documentacion del SMA)
          Fase 2  — Recoleccion de datos: ejecutada por Agente 1 (agente_1_recoleccion.py)
          Fase 3  — Preparacion de datos: ejecutada por Agente 2 (agente_2_preprocesamiento.py)
          Fase 4  — Division del conjunto: particion cronologica dinamica (ultimos 2 anos = test,
                    resto = train); permite reentrenamiento automatico sin cambiar codigo
          Fase 5  — Seleccion del modelo: red LSTM de dos capas apiladas (PyTorch)
                    con lookback=12 meses y 6 variables climaticas/epidemiologicas
          Fase 6a — Entrenamiento baseline LSTM simple (1 capa, hidden=32, lr=0.01, 40 epocas)
          Fase 7a — Evaluacion del baseline (R2, MAE en test set)
          Fase 8  — Optimizacion de hiperparametros: Optuna TPE, 20 trials x K=5 folds
                    cronologicos; sampler TPESampler(seed=9); espacio: hidden_dim,
                    num_layers, lr, dropout
          Fase 6b — Reentrenamiento con mejores hiperparametros + early stopping (max 300 epocas)
                    ReduceLROnPlateau(patience=5) + early stopping patience=15 epocas
          Fase 7b — Evaluacion final (R², MAE, RMSE) sobre el test set
          Fase 9  — Implementacion: serializacion y subida a AWS S3, carga en FastAPI/Railway
          Fase 10 — Mantenimiento: drift detection (PSI sobre features climaticas NASA POWER)
                    + reentrenamiento automatico via GitHub Actions cuando llega nueva version
                    OpenDengue (verificar_actualizacion.py ejecutado el 1ro de cada mes)

        Returns:
            dict con r2_lstm, mae_lstm, rmse_lstm, n_test y lstm_test_lookup
            (predicciones del test set indexadas por (iso_a0, adm_1_name, ano, mes),
            para que el Agente 5 arme el ensemble con las del Agente 3).
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

        # 6 features originales: el LSTM aprende patrones temporales desde datos crudos
        lstm_feats = [
            'tmax_promedio', 'tmin_promedio', 'precipitacion',
            'humedad_promedio', 'agua_basica', 'incidencia_dengue',
        ]
        print(f"   [LSTM] Usando {len(lstm_feats)} features originales")

        X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, lstm_feats, LSTM_SEQ_LEN)

        # ── Fase 4: División cronológica del conjunto (evita data leakage) ──
        # Split dinámico: últimos 2 años = test
        TEST_ANOS  = 2
        max_ano    = int(anos_seq.max())
        split_ano  = max_ano - TEST_ANOS
        train_mask = anos_seq <= split_ano
        test_mask  = anos_seq >  split_ano
        print(f"   [LSTM] Split dinámico: train ≤{split_ano} | test >{split_ano} (max={max_ano})")

        X_train   = X_seq[train_mask]
        y_train   = y_seq[train_mask]
        anos_train = anos_seq[train_mask]
        X_test    = X_seq[test_mask]
        y_test    = y_seq[test_mask]
        test_ids  = [seq_ids[i] for i, m in enumerate(test_mask) if m]

        print(f"   [LSTM] Secuencias — Train: {len(X_train)} | Test: {len(X_test)}")

        # ── Fase 5: Selección del modelo — LSTM 2 capas apiladas (PyTorch) ──
        # Escalador global ajustado sobre todo el train
        escalador = StandardScaler()
        X_train_flat = X_train.reshape(-1, len(lstm_feats))
        escalador.fit(X_train_flat)
        X_train_sc = escalador.transform(X_train_flat).reshape(X_train.shape)
        X_test_sc  = escalador.transform(
            X_test.reshape(-1, len(lstm_feats))).reshape(X_test.shape)
        y_train_log = np.log1p(y_train)

        # ── Fase 6a: Baseline — LSTM simple (1 capa, hidden=32, lr=0.01) ──
        print("\n   [Fase 6a] Baseline LSTM (1 capa, hidden=32, lr=0.01, 40 epocas)...")
        modelo_base = self._entrenar_lstm(X_train_sc, y_train_log,
                                          hidden_dim=32, lr=0.01, dropout=0.0,
                                          epochs=40, seed=self.semilla)

        # ── Fase 7a: Evaluación baseline ──
        r2_base, mae_base = self._evaluar_lstm(modelo_base, X_test_sc, y_test)
        print(f"   [Fase 7a] Baseline — R²={r2_base*100:.2f}%  MAE={mae_base:.4f}")

        # ── Fase 8: Bayesian Optimization (Optuna TPE) + folds cronologicos ──
        N_TRIALS_LSTM  = 20
        anos_unicos_train = sorted(set(anos_train.tolist()))
        fold_val_anos  = anos_unicos_train[-5:]
        folds = [
            (anos_train < val_ano, anos_train == val_ano)
            for val_ano in fold_val_anos
        ]
        print(f"\n   [Fase 8] Optuna TPE — {N_TRIALS_LSTM} trials x K=5 folds cronologicos...")
        print(f"   Folds val: {fold_val_anos}")

        def objective_lstm(trial):
            hidden_dim = trial.suggest_int("hidden_dim", 64, 512, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout    = trial.suggest_float("dropout", 0.0, 0.4)
            r2s = []
            for tr_mask, val_mask in folds:
                if val_mask.sum() == 0:
                    continue
                X_tr = X_train[tr_mask]; y_tr = y_train_log[tr_mask]
                X_vl = X_train[val_mask]; y_vl = y_train[val_mask]
                sc_fold = StandardScaler()
                X_tr_sc = sc_fold.fit_transform(
                    X_tr.reshape(-1, len(lstm_feats))).reshape(X_tr.shape)
                X_vl_sc = sc_fold.transform(
                    X_vl.reshape(-1, len(lstm_feats))).reshape(X_vl.shape)
                m = self._entrenar_lstm(X_tr_sc, y_tr, hidden_dim=hidden_dim,
                                        num_layers=num_layers, lr=lr, dropout=dropout,
                                        epochs=50, seed=self.semilla)
                r2_fold, _ = self._evaluar_lstm(m, X_vl_sc, y_vl)
                r2s.append(r2_fold)
            return float(np.mean(r2s))

        def cb_lstm(study, trial):
            best = " <-- mejor" if trial.value == study.best_value else ""
            p = trial.params
            print(f"  Trial {trial.number+1:02d}  hidden={p['hidden_dim']}  "
                  f"layers={p['num_layers']}  lr={p['lr']:.5f}  "
                  f"dropout={p['dropout']:.2f}  R2_CV={trial.value*100:.2f}%{best}")

        study_lstm = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.semilla)
        )
        study_lstm.optimize(objective_lstm, n_trials=N_TRIALS_LSTM, callbacks=[cb_lstm])
        best_lstm_trial = study_lstm.best_trial
        mejores_params  = best_lstm_trial.params
        print(f"\n   Mejor R2 CV: {best_lstm_trial.value*100:.2f}%")
        for k, v in sorted(mejores_params.items()):
            print(f"     {k:12s}: {v}")

        # ── Fase 6b: Reentrenar con mejores params + early stopping (max 300 épocas) ──
        # Valida en año 2020 con ReduceLROnPlateau (patience=5) y early stopping (patience=15)
        print(f"\n   [Fase 6b] Reentrenando con mejores params (early stopping, max 300 epocas)...")
        ano_val_es = max(set(anos_train.tolist()))   # último año del train como validación ES
        val_es = anos_train == ano_val_es
        fit_es = anos_train <  ano_val_es
        Xf_es = X_train_sc[fit_es]; yf_es = y_train_log[fit_es]
        Xv_es = X_train_sc[val_es]; yv_es = y_train_log[val_es]

        torch.manual_seed(self.semilla); np.random.seed(self.semilla)
        modelo = DengueLSTMModel(len(lstm_feats),
                                  mejores_params['hidden_dim'],
                                  num_layers=mejores_params['num_layers'],
                                  dropout=mejores_params['dropout'])
        opt_es  = optim.Adam(modelo.parameters(), lr=mejores_params['lr'], weight_decay=1e-4)
        sched   = optim.lr_scheduler.ReduceLROnPlateau(opt_es, patience=5, factor=0.5)
        loader_es = DataLoader(
            TensorDataset(torch.tensor(Xf_es, dtype=torch.float32),
                          torch.tensor(yf_es, dtype=torch.float32).unsqueeze(1)),
            batch_size=256, shuffle=True)
        best_val_loss, es_wait, best_state, best_ep = 1e9, 0, None, 0
        for ep in range(300):
            modelo.train()
            for bx, by in loader_es:
                opt_es.zero_grad(); nn.MSELoss()(modelo(bx), by).backward(); opt_es.step()
            modelo.eval()
            with torch.no_grad():
                val_loss = float(nn.MSELoss()(
                    modelo(torch.tensor(Xv_es, dtype=torch.float32)).flatten(),
                    torch.tensor(yv_es, dtype=torch.float32)))
            sched.step(val_loss)
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss; es_wait = 0; best_ep = ep + 1
                best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            else:
                es_wait += 1
                if es_wait >= 15:
                    print(f"   Early stopping en epoch {ep+1} (mejor: epoch {best_ep})")
                    break
        modelo.load_state_dict(best_state)

        # ── Fase 7b: Evaluación final ──
        r2, mae = self._evaluar_lstm(modelo, X_test_sc, y_test)
        modelo.eval()
        with torch.no_grad():
            pred_log_arr = modelo(
                torch.tensor(X_test_sc, dtype=torch.float32)).numpy().flatten()
        pred = np.expm1(pred_log_arr)

        r2_log_lstm = r2_score(np.log1p(y_test), pred_log_arr)
        print(f"   [Fase 7b] Optimizado — R²={r2_log_lstm*100:.2f}%  MAE={mae:.4f}")
        print(f"   Mejora sobre baseline: R² {(r2-r2_base)*100:+.2f}pp  MAE {mae-mae_base:+.4f}")

        # Serializar artefactos localmente
        torch.save(modelo.state_dict(), os.path.join(self.model_dir, "lstm_model.pth"))
        with open(os.path.join(self.model_dir, "escalador_lstm.pkl"), "wb") as f:
            pickle.dump(escalador, f)
        with open(os.path.join(self.model_dir, "lstm_features.pkl"), "wb") as f:
            pickle.dump(lstm_feats, f)

        lstm_config = {
            "seq_len":      LSTM_SEQ_LEN,
            "input_dim":    len(lstm_feats),
            "hidden_dim":   mejores_params['hidden_dim'],
            "num_layers":   mejores_params['num_layers'],
            "lr":           mejores_params['lr'],
            "dropout":      mejores_params['dropout'],
            "r2":           round(float(r2_log_lstm), 4),
            "mae":          round(float(mae), 4),
            "r2_baseline":  round(float(r2_base), 4),
            "r2_cv_mejor":  round(float(best_lstm_trial.value), 4),
            "k_folds":      5,
            "optimizer":    "Optuna/TPE",
            "n_trials":     N_TRIALS_LSTM,
            "trials": [{"trial": t.number, "params": t.params, "r2_cv": round(t.value, 4)}
                       for t in study_lstm.trials],
        }
        with open(os.path.join(self.model_dir, "lstm_config.json"), "w") as f:
            json.dump(lstm_config, f, indent=4)

        rmse_lstm = float(np.sqrt(mean_squared_error(y_test, pred)))
        print(f"   [Fase 7b] RMSE={rmse_lstm:.4f}")

        # Lookup de predicciones LSTM en test: (iso_a0, adm_upper, ano, mes) → pred
        # Mismo formato que xgb_test_lookup del Agente 3 — el Agente 5 alinea
        # ambos por clave común para construir el ensemble.
        lstm_test_lookup = {seq_id: float(p) for seq_id, p in zip(test_ids, pred)}

        # Subir artefactos propios a S3 (el metrics.json final lo arma el Agente 5,
        # combinando este resultado con el del Agente 3)
        for fname in ["lstm_model.pth", "escalador_lstm.pkl", "lstm_features.pkl",
                      "lstm_config.json"]:
            s3.upload(os.path.join(self.model_dir, fname), s3.PREFIX_MODELOS + fname)

        print("SUCCESS: [Agente 4] LSTM entrenado y subido a S3.")
        print("=" * 70)

        return {"r2_lstm": round(r2_log_lstm, 4), "mae_lstm": round(mae, 4),
                "rmse_lstm": round(rmse_lstm, 4), "n_test": len(y_test),
                "lstm_test_lookup": lstm_test_lookup}

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
            num_layers=config.get("num_layers", 2),
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


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agente = AgentePrediccionDL(base_dir=base)
    agente.entrenar_modelo()
