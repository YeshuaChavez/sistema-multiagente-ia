# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 5: Orquestador de Consenso (Ensemble + Alertas)
--------------------------------------------------
Responsabilidad: Unifica las predicciones del Agente 3 (LightGBM) y el Agente 4
(LSTM PyTorch) mediante promedio de ensemble, clasifica el nivel de riesgo
epidemiológico con percentiles históricos calibrados por departamento, y coordina
la respuesta final del sistema multi-agente.
"""

import os
import numpy as np
import pandas as pd


class AgenteOrquestador:
    """
    Agente 5 — Orquestador del Sistema Multi-Agente SMA-ML/DL.

    Flujo de inferencia:
      Agente 3 (LightGBM) ─┐
                             ├─► Ensemble (promedio) ─► Nivel de Riesgo
      Agente 4 (LSTM)     ─┘

    Niveles de alerta calibrados con percentiles históricos:
      Normal     (<p25)
      Vigilancia (p25–p50)
      Alerta     (p50–p90)
      Epidemia   (>p90)
    """

    _NIVELES = {
        "normal":     {"nivel": "Normal",     "codigo": "normal",     "color": "#10b981"},
        "vigilancia": {"nivel": "Vigilancia", "codigo": "vigilancia", "color": "#eab308"},
        "alerta":     {"nivel": "Alerta",     "codigo": "alerta",     "color": "#f97316"},
        "epidemia":   {"nivel": "Epidemia",   "codigo": "epidemia",   "color": "#ef4444"},
    }

    def __init__(self, agente_ml, agente_dl, df_master, df_coords=None):
        """
        Args:
            agente_ml:  AgentePrediccionML cargado en modo inferencia (cargar_modelo).
            agente_dl:  AgentePrediccionDL cargado en modo inferencia (cargar_modelo).
            df_master:  DataFrame maestro mensual (dataset_maestro_mensual_latam.csv).
            df_coords:  DataFrame de coordenadas departamentales (opcional).
        """
        self.agente_ml = agente_ml
        self.agente_dl = agente_dl
        self.df_master = df_master
        self.df_coords = df_coords

        # Percentiles globales como fallback cuando el departamento no tiene historial suficiente
        self.p25 = float(df_master["incidencia_dengue"].quantile(0.25))
        self.p50 = float(df_master["incidencia_dengue"].quantile(0.50))
        self.p90 = float(df_master["incidencia_dengue"].quantile(0.90))

        print(f"   [Agente 5] Orquestador listo — percentiles globales: "
              f"p25={self.p25:.2f}, p50={self.p50:.2f}, p90={self.p90:.2f}")

    # ─────────────────────────────────────────────────────────────
    # CLASIFICACIÓN DE RIESGO EPIDEMIOLÓGICO
    # ─────────────────────────────────────────────────────────────

    def calcular_nivel_riesgo(self, pred_val, iso_a0=None, adm_1_name=None):
        """Clasifica el nivel de riesgo usando percentiles del departamento (o globales)."""
        p25, p50, p90 = self.p25, self.p50, self.p90

        if iso_a0 and adm_1_name:
            df_dept = self.df_master[
                (self.df_master['iso_a0'] == iso_a0.strip().upper()) &
                (self.df_master['adm_1_name'].str.upper() == adm_1_name.strip().upper())
            ]
            if not df_dept.empty:
                p25 = float(df_dept["incidencia_dengue"].quantile(0.25))
                p50 = max(float(df_dept["incidencia_dengue"].quantile(0.50)), 0.5)
                p90 = max(float(df_dept["incidencia_dengue"].quantile(0.90)), 5.0)

        if pred_val <= p25:
            return self._NIVELES["normal"]
        elif pred_val <= p50:
            return self._NIVELES["vigilancia"]
        elif pred_val <= p90:
            return self._NIVELES["alerta"]
        else:
            return self._NIVELES["epidemia"]

    # ─────────────────────────────────────────────────────────────
    # PREDICCIÓN ORQUESTADA (AGENTE 3 + AGENTE 4 + ENSEMBLE)
    # ─────────────────────────────────────────────────────────────

    def predecir_departamento(self, iso_a0, adm_1_name, ano=None, mes=None, clima_overrides=None):
        """
        Orquesta la predicción completa de incidencia de dengue para un departamento.

        Pasos:
          1. Recupera el historial del departamento.
          2. Construye el vector de 34 features para Agente 3 (LightGBM).
          3. Agente 3 → predicción ML + SHAP local.
          4. Agente 4 → predicción LSTM con secuencia de 12 meses.
          5. Ensemble = (ML + LSTM) / 2.
          6. Clasifica nivel de riesgo para cada predicción.

        Returns:
            dict con prediccion_ml, prediccion_lstm, prediccion_ensemble,
            riesgo_*, shap_local, features_usadas, percentiles_locales.
        """
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()

        df_dept = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)

        if df_dept.empty:
            raise ValueError(f"No se encontraron registros para {adm_1_name} ({iso_a0})")

        # ── Registro de referencia ──
        target_row = pd.DataFrame()
        if ano is not None and mes is not None:
            target_row = df_dept[(df_dept['ano'] == ano) & (df_dept['mes'] == mes)]
        if target_row.empty:
            target_row = df_dept.iloc[[-1]]

        base_record = target_row.iloc[0].to_dict()
        ref_idx = list(target_row.index)[0]
        ref_mes = int(base_record.get('mes', 1))

        if clima_overrides:
            for key, val in clima_overrides.items():
                if key in base_record:
                    base_record[key] = float(val)

        # ── Vector de features para Agente 3 ──
        cols_feat = self.agente_ml.cols_feat
        vector = []
        for feat in cols_feat:
            val = base_record.get(feat)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                vector.append(val)
            elif "_lag" in feat:
                parts = feat.split("_lag")
                var_base = parts[0]
                lag_num = int(parts[1])
                lag_val = None
                if ref_idx >= lag_num:
                    map_vars = {
                        "tmax": "tmax_promedio", "tmin": "tmin_promedio",
                        "precipitacion": "precipitacion", "humedad": "humedad_promedio",
                        "incidencia": "incidencia_dengue",
                        "incidencia_vecinos": "incidencia_dengue",
                    }
                    col_real = map_vars.get(var_base, var_base)
                    if col_real in df_dept.columns:
                        lag_val = df_dept.loc[ref_idx - lag_num, col_real]
                if lag_val is None or (isinstance(lag_val, float) and np.isnan(lag_val)):
                    med = df_dept[df_dept['mes'] == ref_mes].median(numeric_only=True).to_dict()
                    lag_val = med.get(var_base, 0.0)
                vector.append(lag_val)
            elif feat == "incidencia_roll3":
                vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 3):ref_idx].values
                vector.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
            elif feat == "incidencia_roll6":
                vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 6):ref_idx].values
                vector.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
            elif feat == "mes_sin":
                vector.append(float(np.sin(2 * np.pi * ref_mes / 12)))
            elif feat == "mes_cos":
                vector.append(float(np.cos(2 * np.pi * ref_mes / 12)))
            else:
                vector.append(0.0)

        if clima_overrides:
            for i, feat in enumerate(cols_feat):
                if feat in clima_overrides:
                    vector[i] = float(clima_overrides[feat])

        # ── Agente 3: LightGBM ──
        res_ml = self.agente_ml.predecir(vector, compute_shap=True)
        pred_ml = res_ml["prediccion_ml"]

        # ── Agente 4: LSTM PyTorch ──
        pred_lstm = self.agente_dl.predecir_secuencia(df_dept, ref_idx, clima_overrides)

        # ── Agente 5: Ensemble ──
        pred_ens = (pred_ml + pred_lstm) / 2.0 if pred_lstm is not None else pred_ml

        result = {
            "prediccion_ml":       round(pred_ml, 4),
            "riesgo_ml":           self.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
            "prediccion_lstm":     round(pred_lstm, 4) if pred_lstm is not None else None,
            "riesgo_lstm":         self.calcular_nivel_riesgo(pred_lstm, iso_a0, adm_1_name) if pred_lstm is not None else None,
            "prediccion_ensemble": round(pred_ens, 4),
            "riesgo_ensemble":     self.calcular_nivel_riesgo(pred_ens, iso_a0, adm_1_name),
            "features_usadas":     {f: float(v) for f, v in zip(cols_feat, vector)},
        }
        if "shap_local" in res_ml:
            result["shap_local"] = res_ml["shap_local"]

        # Percentiles locales del departamento
        df_d = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ]
        p25, p50, p90 = self.p25, self.p50, self.p90
        if not df_d.empty:
            p25 = float(df_d["incidencia_dengue"].quantile(0.25))
            p50 = max(float(df_d["incidencia_dengue"].quantile(0.50)), 0.5)
            p90 = max(float(df_d["incidencia_dengue"].quantile(0.90)), 5.0)
        result["percentiles_locales"] = {
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p90": round(p90, 4),
        }

        return result

    # ─────────────────────────────────────────────────────────────
    # SERVICIOS DE DATOS
    # ─────────────────────────────────────────────────────────────

    def obtener_metadatos_paises(self):
        """Retorna {pais: {iso_a0, departamentos[]}} de todos los países disponibles."""
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
        """Retorna la serie histórica mensual de un departamento."""
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()
        df_f = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)
        records = []
        for r in df_f.itertuples():
            records.append({
                "fecha":         f"{r.ano}-{r.mes:02d}",
                "ano":           int(r.ano),
                "mes":           int(r.mes),
                "casos":         int(r.casos_dengue),
                "incidencia":    float(r.incidencia_dengue),
                "tmax":          float(r.tmax_promedio),
                "tmin":          float(r.tmin_promedio),
                "precipitacion": float(r.precipitacion),
                "humedad":       float(r.humedad_promedio),
            })
        return records

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
                "name":            f"{row['adm_1_name'].title()} ({row['iso_a0']})",
                "adm_1_name":      row['adm_1_name'],
                "iso_a0":          row['iso_a0'],
                "pais":            row['pais'],
                "mean_incidencia": round(float(row['mean_incidencia']), 1),
                "max_incidencia":  round(float(row['max_incidencia']), 1),
                "pct":             round(float(row['mean_incidencia']) / max_mean * 100, 1),
            })
        return result

    def obtener_shap_global(self):
        """Retorna las importancias SHAP globales (Agente 3)."""
        return self.agente_ml.shap_importance
