# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Backend Services
----------------------------
Carga persistente de datos y modelos en RAM.
Ensemble de 3 modelos: LightGBM + MLP PyTorch + LSTM PyTorch.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import lightgbm  # noqa: F401  (import needed for unpickling LGBMRegressor)


class DengueMLPModel(nn.Module):
    def __init__(self, input_dim=34, output_dim=1):
        super(DengueMLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),        nn.ReLU(),
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


class PredictionService:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.model_dir = os.path.join(self.db_dir, "modelos")
        self.processed_dir = os.path.join(self.db_dir, "datos_procesados")
        self.raw_dir = os.path.join(self.db_dir, "datos_crudos")

        self.df_master = None
        self.df_coords = None
        self.cols_feat = None

        # ML (LightGBM)
        self.modelo_ml = None
        self.imputador_ml = None
        self.escalador_ml = None

        # LSTM PyTorch
        self.modelo_lstm = None
        self.escalador_lstm = None
        self.lstm_features = None
        self.lstm_seq_len = 12

        self.shap_importance = None
        self.shap_explainer = None
        self.p25 = self.p50 = self.p90 = 0.0
        self.oni_lookup = {}  # {(ano, mes): {'oni': v, 'oni_lag1': v, ...}}

        self.inicializar_servicio()

    def inicializar_servicio(self):
        print("[In-Memory Service] Iniciando carga de activos en RAM...")

        # 1. Dataset maestro
        master_path = os.path.join(self.processed_dir, "dataset_maestro_mensual_latam.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Falta dataset maestro: {master_path}")
        self.df_master = pd.read_csv(master_path)
        print(f"   -> Dataset cargado: {self.df_master.shape[0]} registros.")

        self.p25 = float(self.df_master["incidencia_dengue"].quantile(0.25))
        self.p50 = float(self.df_master["incidencia_dengue"].quantile(0.50))
        self.p90 = float(self.df_master["incidencia_dengue"].quantile(0.90))

        # 2. Coordenadas
        coords_path = os.path.join(self.raw_dir, "departamentos_coordenadas.csv")
        if os.path.exists(coords_path):
            self.df_coords = pd.read_csv(coords_path)
            self.df_coords['iso_a0'] = self.df_coords['iso_a0'].astype(str).str.strip().str.upper()
            self.df_coords['adm_1_name'] = self.df_coords['adm_1_name'].astype(str).str.strip().str.upper()

        # 3. Lista de features
        cols_feat_path = os.path.join(self.model_dir, "cols_feat.pkl")
        if not os.path.exists(cols_feat_path):
            raise FileNotFoundError(f"Falta cols_feat: {cols_feat_path}")
        with open(cols_feat_path, "rb") as f:
            self.cols_feat = pickle.load(f)

        # 4. LightGBM
        lgbm_path = os.path.join(self.model_dir, "lgbm_model.pkl")
        if not os.path.exists(lgbm_path):
            raise FileNotFoundError(f"Falta lgbm_model.pkl: {lgbm_path}")
        with open(lgbm_path, "rb") as f:
            self.modelo_ml = pickle.load(f)
        with open(os.path.join(self.model_dir, "imputador_ml.pkl"), "rb") as f:
            self.imputador_ml = pickle.load(f)
        with open(os.path.join(self.model_dir, "escalador_ml.pkl"), "rb") as f:
            self.escalador_ml = pickle.load(f)

        # SHAP explainer para LightGBM (TreeSHAP local)
        self.shap_explainer = shap.TreeExplainer(self.modelo_ml)

        # 5. LSTM PyTorch
        lstm_path = os.path.join(self.model_dir, "lstm_model.pth")
        lstm_config_path = os.path.join(self.model_dir, "lstm_config.json")
        lstm_features_path = os.path.join(self.model_dir, "lstm_features.pkl")
        escalador_lstm_path = os.path.join(self.model_dir, "escalador_lstm.pkl")

        if all(os.path.exists(p) for p in [lstm_path, lstm_config_path, lstm_features_path, escalador_lstm_path]):
            with open(lstm_config_path, "r") as f:
                lstm_config = json.load(f)
            with open(lstm_features_path, "rb") as f:
                self.lstm_features = pickle.load(f)
            with open(escalador_lstm_path, "rb") as f:
                self.escalador_lstm = pickle.load(f)

            self.lstm_seq_len = lstm_config.get("seq_len", 12)
            self.modelo_lstm = DengueLSTMModel(
                input_dim=lstm_config.get("input_dim", len(self.lstm_features))
            )
            self.modelo_lstm.load_state_dict(
                torch.load(lstm_path, map_location=torch.device('cpu'))
            )
            self.modelo_lstm.eval()
            print("   -> LSTM cargado.")
        else:
            print("   -> LSTM no disponible (faltan archivos).")

        # 7. ONI / ENSO lookup
        oni_path = os.path.join(self.raw_dir, "oni_mensual.csv")
        if os.path.exists(oni_path):
            df_oni = pd.read_csv(oni_path)
            oni_cols = [c for c in ['oni', 'oni_lag1', 'oni_lag2', 'oni_lag3'] if c in df_oni.columns]
            for _, row in df_oni.iterrows():
                self.oni_lookup[(int(row['ano']), int(row['mes']))] = {
                    c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in oni_cols
                }
            print(f"   -> ONI cargado: {len(self.oni_lookup)} registros.")

        # 8. SHAP global
        shap_path = os.path.join(self.model_dir, "shap_importance.json")
        if os.path.exists(shap_path):
            with open(shap_path, "r") as f:
                self.shap_importance = json.load(f)

        print("SUCCESS: [In-Memory Service] Todos los modelos cargados en RAM.")

    # ─────────────────────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────────────────────

    def calcular_nivel_riesgo(self, pred_val, iso_a0=None, adm_1_name=None):
        p25, p50, p90 = self.p25, self.p50, self.p90

        if iso_a0 and adm_1_name:
            df_dept = self.df_master[
                (self.df_master['iso_a0'] == iso_a0.strip().upper()) &
                (self.df_master['adm_1_name'].str.upper() == adm_1_name.strip().upper())
            ]
            if not df_dept.empty:
                p25 = float(df_dept["incidencia_dengue"].quantile(0.25))
                p50 = float(df_dept["incidencia_dengue"].quantile(0.50))
                p90 = float(df_dept["incidencia_dengue"].quantile(0.90))
                p50 = max(p50, 0.5)
                p90 = max(p90, 5.0)

        if pred_val <= p25:
            return {"nivel": "Normal",     "codigo": "normal",    "color": "#10b981"}
        elif pred_val <= p50:
            return {"nivel": "Vigilancia", "codigo": "vigilancia","color": "#eab308"}
        elif pred_val <= p90:
            return {"nivel": "Alerta",     "codigo": "alerta",    "color": "#f97316"}
        else:
            return {"nivel": "Epidemia",   "codigo": "epidemia",  "color": "#ef4444"}

    def _predict_lstm_sequence(self, df_dept, ref_idx, clima_overrides=None):
        """Construye la secuencia LSTM desde el historial y ejecuta inferencia."""
        if self.modelo_lstm is None or self.escalador_lstm is None:
            return None

        # Tomar las últimas lstm_seq_len filas hasta ref_idx (inclusive)
        start = max(0, ref_idx - self.lstm_seq_len + 1)
        hist = df_dept.iloc[start:ref_idx + 1]
        feat_arr = hist[self.lstm_features].values.copy().astype(float)

        # Pad con ceros si hay menos de lstm_seq_len filas
        if len(feat_arr) < self.lstm_seq_len:
            pad = np.zeros((self.lstm_seq_len - len(feat_arr), len(self.lstm_features)))
            feat_arr = np.vstack([pad, feat_arr])

        # Aplicar overrides al último timestep de la secuencia
        if clima_overrides:
            override_map = {f: i for i, f in enumerate(self.lstm_features)}
            for feat_name, feat_i in override_map.items():
                if feat_name in clima_overrides:
                    feat_arr[-1, feat_i] = float(clima_overrides[feat_name])
            # incidencia_lag1 del predictor → incidencia_dengue en el último timestep de la secuencia
            if 'incidencia_lag1' in clima_overrides:
                inc_idx = self.lstm_features.index('incidencia_dengue')
                feat_arr[-1, inc_idx] = float(clima_overrides['incidencia_lag1'])

        flat = feat_arr.reshape(-1, len(self.lstm_features))
        scaled = self.escalador_lstm.transform(flat).reshape(1, self.lstm_seq_len, len(self.lstm_features))

        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_log = float(self.modelo_lstm(x_tensor).numpy()[0][0])
        return max(0.0, np.expm1(pred_log))

    # ─────────────────────────────────────────────────────────────
    # PREDICCIÓN PRINCIPAL (LightGBM + MLP, sin LSTM)
    # ─────────────────────────────────────────────────────────────

    def realizar_prediccion_vector(self, vector_x, iso_a0=None, adm_1_name=None, compute_shap=False):
        """Predicción LightGBM a partir de un vector de features plano."""
        entrada = pd.DataFrame([vector_x], columns=self.cols_feat)

        entrada_imp_ml = self.imputador_ml.transform(entrada)
        entrada_esc_ml = pd.DataFrame(
            self.escalador_ml.transform(entrada_imp_ml), columns=self.cols_feat
        )
        pred_ml_log = float(self.modelo_ml.predict(entrada_esc_ml)[0])
        pred_ml = max(0.0, np.expm1(pred_ml_log))

        result = {
            "prediccion_ml":       round(pred_ml, 4),
            "riesgo_ml":           self.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
            "prediccion_ensemble": round(pred_ml, 4),
            "riesgo_ensemble":     self.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
        }

        if compute_shap and self.shap_explainer is not None:
            shap_vals = self.shap_explainer.shap_values(entrada_esc_ml)
            # Normalize across SHAP API versions
            if hasattr(shap_vals, 'values'):
                raw_vals = shap_vals.values[0]          # Explanation object (shap >= 0.41)
            elif isinstance(shap_vals, list):
                raw_vals = shap_vals[0][0]              # multi-output list
            else:
                raw_vals = np.asarray(shap_vals)[0]     # standard 2D ndarray
            result["shap_local"] = {
                feat: round(float(val), 6)
                for feat, val in zip(self.cols_feat, raw_vals)
            }

        return result

    def obtener_top_departamentos(self, n=5):
        """Retorna los n departamentos con mayor incidencia media histórica."""
        grp = (
            self.df_master
            .groupby(['adm_1_name', 'iso_a0', 'pais'])['incidencia_dengue']
            .agg(mean_incidencia='mean', max_incidencia='max')
            .reset_index()
            .nlargest(n, 'mean_incidencia')
        )
        max_mean = float(grp['mean_incidencia'].max()) or 1.0
        result = []
        for _, row in grp.iterrows():
            result.append({
                "name": f"{row['adm_1_name'].title()} ({row['iso_a0']})",
                "adm_1_name": row['adm_1_name'],
                "iso_a0": row['iso_a0'],
                "pais": row['pais'],
                "mean_incidencia": round(float(row['mean_incidencia']), 1),
                "max_incidencia": round(float(row['max_incidencia']), 1),
                "pct": round(float(row['mean_incidencia']) / max_mean * 100, 1),
            })
        return result

    # ─────────────────────────────────────────────────────────────
    # SIMULACIÓN COMPLETA (LightGBM + MLP + LSTM = Ensemble 3-way)
    # ─────────────────────────────────────────────────────────────

    def simular_prediccion_departamento(self, iso_a0, adm_1_name, ano=None, mes=None, clima_overrides=None):
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()

        df_dept = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)

        if df_dept.empty:
            raise ValueError(f"No se encontraron registros para {adm_1_name} ({iso_a0})")

        # Determinar registro de referencia
        target_row = pd.DataFrame()
        if ano is not None and mes is not None:
            target_row = df_dept[(df_dept['ano'] == ano) & (df_dept['mes'] == mes)]
        if target_row.empty:
            target_row = df_dept.iloc[[-1]]

        base_record = target_row.iloc[0].to_dict()
        idx_target = list(target_row.index)
        ref_idx = idx_target[0]
        ref_mes = int(base_record.get('mes', 1))

        if clima_overrides:
            for key, val in clima_overrides.items():
                if key in base_record:
                    base_record[key] = float(val)

        # ─── Construir vector de features para LightGBM + MLP ───
        vector = []
        for feat in self.cols_feat:
            if feat in base_record:
                vector.append(base_record[feat])
            elif "_lag" in feat:
                parts = feat.split("_lag")
                var_base = parts[0]
                lag_num = int(parts[1])
                val = None
                if idx_target and ref_idx >= lag_num:
                    map_vars = {
                        "tmax": "tmax_promedio", "tmin": "tmin_promedio",
                        "precipitacion": "precipitacion", "humedad": "humedad_promedio",
                        "incidencia": "incidencia_dengue",
                        "incidencia_vecinos": "incidencia_dengue",
                    }
                    col_real = map_vars.get(var_base, var_base)
                    if col_real in df_dept.columns:
                        val = df_dept.loc[ref_idx - lag_num, col_real]
                if val is None or pd.isna(val):
                    val_med = df_dept[df_dept['mes'] == ref_mes].median(numeric_only=True).to_dict()
                    val = val_med.get(var_base, 0.0)
                vector.append(val)
            elif feat == "incidencia_roll3":
                roll_vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 3):ref_idx].values
                vector.append(float(np.mean(roll_vals)) if len(roll_vals) > 0 else 0.0)
            elif feat == "incidencia_roll6":
                roll_vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 6):ref_idx].values
                vector.append(float(np.mean(roll_vals)) if len(roll_vals) > 0 else 0.0)
            elif feat == "mes_sin":
                vector.append(float(np.sin(2 * np.pi * ref_mes / 12)))
            elif feat == "mes_cos":
                vector.append(float(np.cos(2 * np.pi * ref_mes / 12)))
            elif feat in ('oni', 'oni_lag1', 'oni_lag2', 'oni_lag3'):
                ref_ano = int(base_record.get('ano', 2022))
                if feat == 'oni':
                    oni_val = self.oni_lookup.get((ref_ano, ref_mes), {}).get('oni', 0.0)
                else:
                    lag_num = int(feat.replace('oni_lag', ''))
                    import datetime
                    t = datetime.date(ref_ano, ref_mes, 1)
                    for _ in range(lag_num):
                        if t.month == 1:
                            t = t.replace(year=t.year - 1, month=12)
                        else:
                            t = t.replace(month=t.month - 1)
                    oni_val = self.oni_lookup.get((t.year, t.month), {}).get(feat, 0.0)
                vector.append(oni_val)
            else:
                vector.append(0.0)

        # Segunda pasada: aplicar overrides directamente por nombre de feature
        if clima_overrides:
            for i, feat in enumerate(self.cols_feat):
                if feat in clima_overrides:
                    vector[i] = float(clima_overrides[feat])

        # ─── LightGBM (con SHAP local) ───
        res = self.realizar_prediccion_vector(vector, iso_a0, adm_1_name, compute_shap=True)

        # ─── LSTM (secuencia temporal) ───
        pred_lstm = self._predict_lstm_sequence(df_dept, ref_idx, clima_overrides)

        if pred_lstm is not None:
            # Ensemble 2-way: LightGBM + LSTM (MLP excluido por menor R²)
            pred_ens2 = (res["prediccion_ml"] + pred_lstm) / 2.0
            res["prediccion_lstm"] = round(pred_lstm, 4)
            res["riesgo_lstm"] = self.calcular_nivel_riesgo(pred_lstm, iso_a0, adm_1_name)
            res["prediccion_ensemble"] = round(pred_ens2, 4)
            res["riesgo_ensemble"] = self.calcular_nivel_riesgo(pred_ens2, iso_a0, adm_1_name)
        else:
            res["prediccion_lstm"] = None
            res["riesgo_lstm"] = None

        res["features_usadas"] = {feat: float(val) for feat, val in zip(self.cols_feat, vector)}

        # Percentiles locales
        df_d = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ]
        p25, p50, p90 = self.p25, self.p50, self.p90
        if not df_d.empty:
            p25 = float(df_d["incidencia_dengue"].quantile(0.25))
            p50 = max(float(df_d["incidencia_dengue"].quantile(0.50)), 0.5)
            p90 = max(float(df_d["incidencia_dengue"].quantile(0.90)), 5.0)
        res["percentiles_locales"] = {"p25": round(p25, 4), "p50": round(p50, 4), "p90": round(p90, 4)}

        return res

    # ─────────────────────────────────────────────────────────────
    # METADATOS Y SERIES HISTÓRICAS
    # ─────────────────────────────────────────────────────────────

    def obtener_metadatos_paises(self):
        paises_dict = {}
        for row in self.df_master[['pais', 'iso_a0', 'adm_1_name']].drop_duplicates().itertuples():
            if row.pais not in paises_dict:
                paises_dict[row.pais] = {"iso_a0": row.iso_a0, "departamentos": []}
            if row.adm_1_name not in paises_dict[row.pais]["departamentos"]:
                paises_dict[row.pais]["departamentos"].append(row.adm_1_name)
        for p in paises_dict:
            paises_dict[p]["departamentos"].sort()
        return paises_dict

    def obtener_historico_departamento(self, iso_a0, adm_1_name):
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()
        df_f = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)
        records = []
        for r in df_f.itertuples():
            records.append({
                "fecha": f"{r.ano}-{r.mes:02d}",
                "ano": int(r.ano), "mes": int(r.mes),
                "casos": int(r.casos_dengue),
                "incidencia": float(r.incidencia_dengue),
                "tmax": float(r.tmax_promedio), "tmin": float(r.tmin_promedio),
                "precipitacion": float(r.precipitacion),
                "humedad": float(r.humedad_promedio)
            })
        return records
