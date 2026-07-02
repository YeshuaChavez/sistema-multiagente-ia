# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 5: Orquestador de Consenso — "Semáforo" (Ensemble + Alertas)
--------------------------------------------------
Responsabilidad: Único punto del sistema que combina los resultados del
Agente 3 (XGBoost, solo ML) y el Agente 4 (LSTM, solo DL), que entrenan de
forma independiente entre sí. En entrenamiento (generar_metricas_finales),
arma el ensemble con pesos base fijos 0.5/0.5 y escribe el metrics.json
final. En inferencia, unifica las predicciones en tiempo real, clasifica el
nivel de riesgo epidemiológico con percentiles históricos calibrados por
departamento, y coordina la respuesta final del sistema multi-agente. El
Agente 6 es quien ajusta los pesos 0.5/0.5 dinámicamente en cada inferencia.

Ciclo de vida ML/DL — este agente es el único que cruza dos fases distintas:
  - Fase 7b (Evaluación final): generar_metricas_finales() calcula el R²/MAE/
    RMSE honesto del ensemble sobre las filas comunes del test set, una vez
    que Agente 3 y Agente 4 ya terminaron su propia Fase 7b por separado.
  - Fase 9 (Implementación): predecir_departamento() sirve inferencia online
    en producción, delegando en el Agente 6 el ajuste dinámico de pesos.
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, cohen_kappa_score, classification_report)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3


class AgenteOrquestador:
    """
    Agente 5 — Orquestador del Sistema Multi-Agente SMA-ML/DL.

    Flujo de inferencia:
      Agente 3 (XGBoost)  ─┐
                             ├─► Ensemble (promedio) ─► Nivel de Riesgo
      Agente 4 (LSTM)     ─┘

    Niveles de alerta calibrados con percentiles históricos:
      Endémico   (<=p50)
      Alerta     (p50–p90)
      Epidemia   (>p90)
    """

    _NIVELES = {
        "endemico": {"nivel": "Endémico", "codigo": "endemico", "color": "#10b981"},
        "alerta":   {"nivel": "Alerta",   "codigo": "alerta",   "color": "#f97316"},
        "epidemia": {"nivel": "Epidemia", "codigo": "epidemia", "color": "#ef4444"},
    }

    def __init__(self, agente_ml, agente_dl, df_master, df_coords=None, metrics=None):
        """
        Args:
            agente_ml:  AgentePrediccionML cargado en modo inferencia (cargar_modelo).
            agente_dl:  AgentePrediccionDL cargado en modo inferencia (cargar_modelo).
            df_master:  DataFrame maestro mensual (dataset_maestro_mensual_latam.csv).
            df_coords:  DataFrame de coordenadas departamentales (opcional).
            metrics:    Dict con métricas de entrenamiento, incluye ensemble_w_xgb/lstm (opcional).
        """
        self.agente_ml = agente_ml
        self.agente_dl = agente_dl
        self.df_master = df_master
        self.df_coords = df_coords

        # Pesos base del ensemble: cargados desde metrics.json si están disponibles
        self._w_xgb  = float((metrics or {}).get("ensemble_w_xgb",  0.5))
        self._w_lstm = float((metrics or {}).get("ensemble_w_lstm", 0.5))

        # Agente 6: Detección de Régimen Epidémico — cargado por ruta para evitar problemas de sys.path
        _here = os.path.dirname(os.path.abspath(__file__))
        _a6_path = os.path.join(_here, "agente_6_regimen.py")
        _spec = importlib.util.spec_from_file_location("agente_6_regimen", _a6_path)
        _mod  = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        self.agente_regimen = _mod.AgenteRegimen(self._w_xgb, self._w_lstm)

        # Percentiles globales como fallback cuando el departamento no tiene historial suficiente
        self.p25 = float(df_master["incidencia_dengue"].quantile(0.25))
        self.p50 = float(df_master["incidencia_dengue"].quantile(0.50))
        self.p90 = float(df_master["incidencia_dengue"].quantile(0.90))

        # Mapa de vecinos espaciales: (iso_a0, adm_upper) → [3 adm_upper más cercanos]
        self._neighbor_map = {}
        if df_coords is not None:
            df_c = df_coords.copy()
            df_c['iso_a0']     = df_c['iso_a0'].astype(str).str.strip().str.upper()
            df_c['adm_1_name'] = df_c['adm_1_name'].astype(str).str.strip().str.upper()
            for country in df_c['iso_a0'].unique():
                cc = df_c[df_c['iso_a0'] == country]
                depts  = cc['adm_1_name'].values
                coords = cc[['lat', 'lon']].values
                for i, d_i in enumerate(depts):
                    dists = sorted(
                        [(depts[j], np.sqrt((coords[i, 0] - coords[j, 0])**2 +
                                            (coords[i, 1] - coords[j, 1])**2))
                         for j in range(len(depts)) if j != i],
                        key=lambda x: x[1]
                    )
                    self._neighbor_map[(country, d_i)] = [d[0] for d in dists[:3]]

        # Tabla de lookup: (iso_a0, adm_upper, ano, mes) → incidencia_dengue
        df_lu = df_master.copy()
        df_lu['iso_u'] = df_lu['iso_a0'].astype(str).str.strip().str.upper()
        df_lu['adm_u'] = df_lu['adm_1_name'].astype(str).str.strip().str.upper()
        self._inc_lookup = {
            (r.iso_u, r.adm_u, int(r.ano), int(r.mes)): float(r.incidencia_dengue)
            for r in df_lu.itertuples()
        }

        print(f"   [Agente 5] Orquestador listo — percentiles globales: "
              f"p25={self.p25:.2f}, p50={self.p50:.2f}, p90={self.p90:.2f}")
        print(f"   [Agente 5] Vecinos espaciales pre-computados para "
              f"{len(self._neighbor_map)} departamentos.")

    # ─────────────────────────────────────────────────────────────
    # FASE 7b — EVALUACIÓN FINAL: CONSOLIDACIÓN DE MÉTRICAS
    # (post Agente 3 + Agente 4, cada uno ya evaluado por separado)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def generar_metricas_finales(metricas_ml, metricas_dl, base_dir):
        """
        "Semáforo" del sistema: se ejecuta una vez que el Agente 3 (XGBoost) y
        el Agente 4 (LSTM) terminaron de entrenar por separado. Alinea sus
        predicciones de test por clave común (iso_a0, adm_1_name, ano, mes),
        arma el ensemble con pesos fijos w_xgb=w_lstm=0.5, clasifica en 3
        niveles de riesgo (Endémico/Alerta/Epidemia) usando percentiles
        históricos por departamento, y escribe/sube el metrics.json final
        consumido por el endpoint /api/metrics.

        Pesos fijos 0.5/0.5: se evaluó ponderar proporcional al R² individual
        de cada modelo, pero XGBoost y LSTM rinden casi igual (91.49% vs
        90.35% R²) y la ganancia era despreciable frente a la simplicidad de
        pesos iguales. El Agente 6 ajusta w_xgb/w_lstm de forma dinámica en
        inferencia según el régimen epidémico — este método solo fija los
        pesos *base* que el Agente 6 recibe como punto de partida.

        Args:
            metricas_ml: dict retornado por AgentePrediccionML.entrenar_modelo()
                         (r2_xgb, mae_xgb, rmse_xgb, n_train, n_test, xgb_test_lookup).
            metricas_dl: dict retornado por AgentePrediccionDL.entrenar_modelo()
                         (r2_lstm, mae_lstm, rmse_lstm, n_test, lstm_test_lookup).
            base_dir:    directorio base del proyecto.

        Returns:
            dict con las métricas finales (el mismo contenido escrito en metrics.json).
        """
        print("=" * 70)
        print("  AGENTE 5 — Consolidando métricas finales (ensemble + clasificación)")
        print("=" * 70)

        model_dir = os.path.join(base_dir, "data", "models")
        feat_path = os.path.join(base_dir, "data", "processed", "dataset_features_latam.csv")

        df = pd.read_csv(feat_path)
        yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly > 100].reset_index(drop=True)

        TEST_ANOS = 2
        split_ano = int(df['ano'].max()) - TEST_ANOS

        # Lookup de incidencia real: (iso_a0, adm_upper, ano, mes) → incidencia_dengue
        df_l = df.copy()
        df_l['iso_u'] = df_l['iso_a0'].astype(str).str.strip().str.upper()
        df_l['adm_u'] = df_l['adm_1_name'].astype(str).str.strip().str.upper()
        y_lookup = {
            (r.iso_u, r.adm_u, int(r.ano), int(r.mes)): float(r.incidencia_dengue)
            for r in df_l.itertuples()
        }

        xgb_lookup  = metricas_ml.get("xgb_test_lookup", {})
        lstm_lookup = metricas_dl.get("lstm_test_lookup", {})
        claves_comunes = sorted(set(xgb_lookup) & set(lstm_lookup) & set(y_lookup))

        W_XGB_FIJO = 0.5
        w_xgb = W_XGB_FIJO
        cls_metrics = {}

        if len(claves_comunes) >= 10:
            xgb_arr  = np.array([xgb_lookup[k]  for k in claves_comunes])
            lstm_arr = np.array([lstm_lookup[k] for k in claves_comunes])
            y_arr    = np.array([y_lookup[k]    for k in claves_comunes])
            lxgb = np.log1p(xgb_arr); llst = np.log1p(lstm_arr); ly = np.log1p(y_arr)
            ens_log  = w_xgb * lxgb + (1.0 - w_xgb) * llst
            ens_raw  = np.expm1(ens_log)
            r2_ens   = r2_score(ly, ens_log)
            mae_ens  = mean_absolute_error(y_arr, ens_raw)
            rmse_ens = float(np.sqrt(mean_squared_error(y_arr, ens_raw)))
            print(f"   [Ensemble] w_xgb={w_xgb:.3f}  w_lstm={1.0 - w_xgb:.3f}  "
                  f"R²={r2_ens*100:.2f}%  MAE={mae_ens:.4f}  RMSE={rmse_ens:.4f}  "
                  f"(n={len(claves_comunes)})")

            # ── Clasificación 3 niveles (Endémico/Alerta/Epidemia) sobre el ensemble ──
            # Percentiles p50/p90 históricos por departamento calculados solo con
            # train (<=split_ano), con piso mínimo y fallback a percentiles globales
            # cuando el departamento no tiene historial suficiente (misma lógica que
            # calcular_nivel_riesgo usa en inferencia).
            df_hist = df[df['ano'] <= split_ano]
            dept_pct = {}
            for (iso, adm), grp in df_hist.groupby(['iso_a0', 'adm_1_name']):
                inc = grp['incidencia_dengue'].values
                dept_pct[(str(iso).upper(), str(adm).upper())] = (
                    max(float(np.percentile(inc, 50)), 0.5),
                    max(float(np.percentile(inc, 90)), 5.0),
                )
            p50_g = max(float(df_hist['incidencia_dengue'].quantile(0.50)), 0.5)
            p90_g = max(float(df_hist['incidencia_dengue'].quantile(0.90)), 5.0)

            def _clasificar(val, iso, adm):
                p50, p90 = dept_pct.get((iso, adm), (p50_g, p90_g))
                if val <= p50:   return 0
                elif val <= p90: return 1
                else:            return 2

            y_true_cls = np.array([_clasificar(v, k[0], k[1]) for v, k in zip(y_arr, claves_comunes)])
            y_pred_cls = np.array([_clasificar(v, k[0], k[1]) for v, k in zip(ens_raw, claves_comunes)])

            acc    = accuracy_score(y_true_cls, y_pred_cls)
            kappa  = cohen_kappa_score(y_true_cls, y_pred_cls)
            report = classification_report(
                y_true_cls, y_pred_cls, labels=[0, 1, 2],
                target_names=["Endemico", "Alerta", "Epidemia"],
                output_dict=True, zero_division=0,
            )
            dist = np.bincount(y_true_cls, minlength=3)
            print(f"   [Clasificación] Acc={acc*100:.2f}%  Kappa={kappa:.4f}")

            cls_metrics = {
                "acc_clasificacion":   round(float(acc), 4),
                "kappa_clasificacion": round(float(kappa), 4),
                "f1_endemico":  round(float(report["Endemico"]["f1-score"]), 4),
                "f1_alerta":    round(float(report["Alerta"]["f1-score"]), 4),
                "f1_epidemia":  round(float(report["Epidemia"]["f1-score"]), 4),
                "precision_endemico": round(float(report["Endemico"]["precision"]), 4),
                "precision_alerta":   round(float(report["Alerta"]["precision"]), 4),
                "precision_epidemia": round(float(report["Epidemia"]["precision"]), 4),
                "recall_endemico": round(float(report["Endemico"]["recall"]), 4),
                "recall_alerta":   round(float(report["Alerta"]["recall"]), 4),
                "recall_epidemia": round(float(report["Epidemia"]["recall"]), 4),
                "soporte_endemico": int(dist[0]),
                "soporte_alerta":   int(dist[1]),
                "soporte_epidemia": int(dist[2]),
            }
        else:
            r2_ens   = (metricas_ml.get("r2_xgb", 0.0)   + metricas_dl.get("r2_lstm", 0.0))   / 2
            mae_ens  = (metricas_ml.get("mae_xgb", 0.0)  + metricas_dl.get("mae_lstm", 0.0))  / 2
            rmse_ens = (metricas_ml.get("rmse_xgb", 0.0) + metricas_dl.get("rmse_lstm", 0.0)) / 2
            print("   [Ensemble] Fallback: pocas filas comunes, promedio simple (pesos igual 0.5/0.5)")

        metrics = {
            "records_procesados": int(len(df)),
            "n_train": int(metricas_ml.get("n_train", 0)),
            "n_test":  int(metricas_ml.get("n_test", 0)),
            "n_paises": int(df['pais'].nunique()),
            "n_departamentos": int(df['adm_1_name'].nunique()),
            "r2_xgb":   metricas_ml.get("r2_xgb", 0.0),
            "mae_xgb":  metricas_ml.get("mae_xgb", 0.0),
            "rmse_xgb": metricas_ml.get("rmse_xgb", 0.0),
            "r2_lstm":   metricas_dl.get("r2_lstm", 0.0),
            "mae_lstm":  metricas_dl.get("mae_lstm", 0.0),
            "rmse_lstm": metricas_dl.get("rmse_lstm", 0.0),
            "r2_ensemble":     round(float(r2_ens), 4),
            "mae_ensemble":    round(float(mae_ens), 4),
            "rmse_ensemble":   round(float(rmse_ens), 4),
            "ensemble_w_xgb":  round(w_xgb, 4),
            "ensemble_w_lstm": round(1.0 - w_xgb, 4),
        }
        metrics.update(cls_metrics)

        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        s3.upload(metrics_path, s3.PREFIX_MODELOS + "metrics.json")

        print("SUCCESS: [Agente 5] metrics.json final generado y subido a S3.")
        print("=" * 70)
        return metrics

    # ─────────────────────────────────────────────────────────────
    # FASE 9 — CLASIFICACIÓN DE RIESGO EPIDEMIOLÓGICO (inferencia online)
    # ─────────────────────────────────────────────────────────────

    def calcular_nivel_riesgo(self, pred_val, iso_a0=None, adm_1_name=None):
        """Clasifica el nivel de riesgo usando percentiles del departamento (o globales)."""
        p50, p90 = self.p50, self.p90

        if iso_a0 and adm_1_name:
            df_dept = self.df_master[
                (self.df_master['iso_a0'] == iso_a0.strip().upper()) &
                (self.df_master['adm_1_name'].str.upper() == adm_1_name.strip().upper())
            ]
            if not df_dept.empty:
                p50 = max(float(df_dept["incidencia_dengue"].quantile(0.50)), 0.5)
                p90 = max(float(df_dept["incidencia_dengue"].quantile(0.90)), 5.0)

        if pred_val <= p50:
            return self._NIVELES["endemico"]
        elif pred_val <= p90:
            return self._NIVELES["alerta"]
        else:
            return self._NIVELES["epidemia"]

    # ─────────────────────────────────────────────────────────────
    # FASE 9 — IMPLEMENTACIÓN: PREDICCIÓN ORQUESTADA
    # (AGENTE 3 + AGENTE 4 + AGENTE 6 + ENSEMBLE, en tiempo real)
    # ─────────────────────────────────────────────────────────────

    def predecir_departamento(self, iso_a0, adm_1_name, ano=None, mes=None, clima_overrides=None, compute_shap=False):
        """
        Orquesta la predicción completa de incidencia de dengue para un departamento.
        Fase 9 del ciclo de vida ML/DL (Implementación) — corre en cada
        request online del backend, no durante el entrenamiento.

        Pasos:
          1. Recupera el historial del departamento.
          2. Construye el vector de 73 features para Agente 3 (XGBoost).
          3. Agente 3 → predicción ML + SHAP local.
          4. Agente 4 → predicción LSTM con secuencia de 12 meses.
          5. Agente 6 → detecta el régimen epidémico y ajusta w_xgb/w_lstm
             a partir de los pesos base 0.5/0.5.
          6. Ensemble = w_xgb_adj × ML + w_lstm_adj × LSTM (pesos del paso 5,
             no un simple promedio).
          7. Clasifica nivel de riesgo para cada predicción.

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

        # Si el usuario especificó un mes objetivo, usarlo para mes_sin/mes_cos
        # aunque no exista ese mes en el CSV (ej: predecir julio 2026)
        if mes is not None:
            ref_mes = int(mes)

        if clima_overrides:
            for key, val in clima_overrides.items():
                if key in base_record:
                    base_record[key] = float(val)

        # ── Vector de features para Agente 3 ──
        cols_feat = self.agente_ml.cols_feat
        # Estas features se recalculan siempre; no deben venir del CSV ni de clima_overrides
        _ALWAYS_COMPUTED = {"mes_sin", "mes_cos", "incidencia_roll3", "incidencia_roll6"}
        # Pre-computar lags de incidencia en log1p (igual que agente_2 al generar features)
        _inc_vars = {"incidencia", "incidencia_vecinos"}

        vector = []
        lag_cache = {}  # {feat: valor_log1p} para reutilizar en features derivadas
        for feat in cols_feat:
            val = None if feat in _ALWAYS_COMPUTED else base_record.get(feat)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                vector.append(val)
            elif "_lag" in feat:
                parts = feat.split("_lag")
                var_base = parts[0]
                lag_num = int(parts[1])
                lag_val = None
                if ref_idx >= lag_num:
                    if var_base == "incidencia_vecinos":
                        lag_row  = df_dept.iloc[ref_idx - lag_num]
                        lag_ano  = int(lag_row['ano'])
                        lag_mes  = int(lag_row['mes'])
                        nbrs = self._neighbor_map.get((iso_a0, adm_1_name_u), [])
                        vals = [self._inc_lookup.get((iso_a0, n, lag_ano, lag_mes))
                                for n in nbrs]
                        vals = [v for v in vals if v is not None and not np.isnan(v)]
                        if vals:
                            lag_val = float(np.mean(vals))
                        else:
                            lag_val = self._inc_lookup.get(
                                (iso_a0, adm_1_name_u, lag_ano, lag_mes))
                        if lag_val is not None:
                            lag_val = np.log1p(lag_val)  # mismo que agente_2 línea 237
                    else:
                        map_vars = {
                            "tmax": "tmax_promedio", "tmin": "tmin_promedio",
                            "precipitacion": "precipitacion", "humedad": "humedad_promedio",
                            "incidencia": "incidencia_dengue",
                        }
                        col_real = map_vars.get(var_base, var_base)
                        if col_real in df_dept.columns:
                            lag_val = df_dept.loc[ref_idx - lag_num, col_real]
                            if var_base == "incidencia" and lag_val is not None:
                                lag_val = np.log1p(lag_val)  # mismo que agente_2 línea 154
                if lag_val is None or (isinstance(lag_val, float) and np.isnan(lag_val)):
                    med = df_dept[df_dept['mes'] == ref_mes].median(numeric_only=True).to_dict()
                    lag_val = med.get(var_base, 0.0)
                    if var_base in _inc_vars:
                        lag_val = np.log1p(lag_val)
                lag_cache[feat] = lag_val
                vector.append(lag_val)
            elif feat == "incidencia_roll3":
                vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 3):ref_idx].values
                v = float(np.log1p(np.mean(vals))) if len(vals) > 0 else 0.0
                lag_cache[feat] = v
                vector.append(v)
            elif feat == "incidencia_roll6":
                vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 6):ref_idx].values
                v = float(np.log1p(np.mean(vals))) if len(vals) > 0 else 0.0
                lag_cache[feat] = v
                vector.append(v)
            elif feat == "incidencia_roll12":
                vals = df_dept['incidencia_dengue'].iloc[max(0, ref_idx - 12):ref_idx].values
                v = float(np.log1p(np.mean(vals))) if len(vals) > 0 else 0.0
                vector.append(v)
            elif feat == "mes_sin":
                vector.append(float(np.sin(2 * np.pi * ref_mes / 12)))
            elif feat == "mes_cos":
                vector.append(float(np.cos(2 * np.pi * ref_mes / 12)))
            # Features derivadas de incidencia (calculadas igual que agente_2)
            elif feat == "aceleracion_incidencia":
                l1 = lag_cache.get("incidencia_lag1", 0.0)
                l2 = lag_cache.get("incidencia_lag2", 0.0)
                vector.append(l1 - l2)
            elif feat == "cambio_interanual":
                l1  = lag_cache.get("incidencia_lag1", 0.0)
                l12 = lag_cache.get("incidencia_lag12", 0.0)
                vector.append(l1 - l12)
            elif feat == "tendencia_1m":
                l1 = lag_cache.get("incidencia_lag1", 0.0)
                l2 = lag_cache.get("incidencia_lag2", 0.0)
                vector.append(np.log1p(l1) - np.log1p(l2))
            elif feat == "tendencia_3m":
                l1 = lag_cache.get("incidencia_lag1", 0.0)
                l3 = lag_cache.get("incidencia_lag3", 0.0)
                vector.append(np.log1p(l1) - np.log1p(l3))
            elif feat == "fase_ascendente":
                l1 = lag_cache.get("incidencia_lag1", 0.0)
                l3 = lag_cache.get("incidencia_lag3", 0.0)
                vector.append(1.0 if l1 > l3 else 0.0)
            elif feat == "indicador_brote":
                l1 = lag_cache.get("incidencia_lag1", 0.0)
                p75 = float(df_dept["incidencia_dengue"].apply(np.log1p).quantile(0.75))
                vector.append(1.0 if l1 > p75 else 0.0)
            # Features derivadas de clima
            elif feat == "amplitud_termica":
                tmax = base_record.get("tmax_promedio", 0.0) or 0.0
                tmin = base_record.get("tmin_promedio", 0.0) or 0.0
                vector.append(float(tmax - tmin))
            elif feat == "temperatura_media":
                tmax = base_record.get("tmax_promedio", 0.0) or 0.0
                tmin = base_record.get("tmin_promedio", 0.0) or 0.0
                vector.append(float((tmax + tmin) / 2))
            elif feat == "precipitacion_anomalia":
                prec = base_record.get("precipitacion", 0.0) or 0.0
                med_prec = df_dept[df_dept['mes'] == ref_mes]['precipitacion'].median()
                vector.append(float(prec - (med_prec if not np.isnan(med_prec) else 0.0)))
            else:
                vector.append(0.0)

        # Incidencia features que el usuario puede ajustar — siempre en log1p (igual que entrenamiento)
        _INC_LAG_FEATS = {f for f in cols_feat if f.startswith("incidencia_lag") or f.startswith("incidencia_vecinos_lag") or f in ("incidencia_roll3", "incidencia_roll6", "incidencia_roll12")}
        if clima_overrides:
            for i, feat in enumerate(cols_feat):
                if feat in clima_overrides and feat not in _ALWAYS_COMPUTED:
                    raw_val = float(clima_overrides[feat])
                    vector[i] = np.log1p(raw_val) if feat in _INC_LAG_FEATS else raw_val
            # Sincronizar lag_cache con los overrides del usuario para que Agente 6
            # use los valores simulados, no los históricos originales
            for feat in ("incidencia_lag1", "incidencia_lag2", "incidencia_lag3"):
                if feat in clima_overrides:
                    lag_cache[feat] = np.log1p(float(clima_overrides[feat]))

        # ── Agente 3: XGBoost ──
        res_ml = self.agente_ml.predecir(vector, compute_shap=compute_shap)
        pred_ml = res_ml["prediccion_ml"]

        # ── Agente 4: LSTM PyTorch ──
        pred_lstm = self.agente_dl.predecir_secuencia(df_dept, ref_idx, clima_overrides)

        # ── Agente 6: Detección de Régimen Epidémico ──
        df_d_pct  = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ]
        p25_local = float(df_d_pct["incidencia_dengue"].quantile(0.25)) if not df_d_pct.empty else self.p25
        p50_local = float(df_d_pct["incidencia_dengue"].quantile(0.50)) if not df_d_pct.empty else self.p50
        # p90 usa el global como piso — evita clasificar como extremo valores bajos en abs.
        # en depts. de baja endemia el p90 local puede ser 2-3 casos/100k
        p90_local = max(
            float(df_d_pct["incidencia_dengue"].quantile(0.90)) if not df_d_pct.empty else self.p90,
            self.p90
        )

        lag1_raw = np.expm1(lag_cache.get("incidencia_lag1", 0.0))
        lag1_log = lag_cache.get("incidencia_lag1", 0.0)
        lag2_log = lag_cache.get("incidencia_lag2", 0.0)

        regimen_info = self.agente_regimen.detectar(
            lag1_raw, lag1_log, lag2_log,
            p25_local, p50_local, p90_local
        )
        w_xgb_adj = regimen_info["w_xgb"]
        w_lstm_adj = regimen_info["w_lstm"]

        # ── Agente 5: Ensemble con pesos del Agente 6 ──
        if pred_lstm is not None:
            pred_ens = w_xgb_adj * pred_ml + w_lstm_adj * pred_lstm
        else:
            pred_ens = pred_ml

        result = {
            "prediccion_ml":       round(pred_ml, 4),
            "riesgo_ml":           self.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
            "prediccion_lstm":     round(pred_lstm, 4) if pred_lstm is not None else None,
            "riesgo_lstm":         self.calcular_nivel_riesgo(pred_lstm, iso_a0, adm_1_name) if pred_lstm is not None else None,
            "prediccion_ensemble": round(pred_ens, 4),
            "riesgo_ensemble":     self.calcular_nivel_riesgo(pred_ens, iso_a0, adm_1_name),
            "ensemble_w_xgb":      round(w_xgb_adj, 4),
            "ensemble_w_lstm":     round(w_lstm_adj, 4),
            "regimen_epidemico":   regimen_info["regimen"],
            "regimen_descripcion": regimen_info["descripcion"],
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
    # FEATURES SIN INFERENCIA (para pre-carga de sliders)
    # ─────────────────────────────────────────────────────────────

    def obtener_features_departamento(self, iso_a0, adm_1_name, ano=None, mes=None):
        """
        Fase 9 (Implementación) — variante de solo lectura de
        predecir_departamento(): construye el vector de 73 features para un
        departamento sin correr ningún modelo (ni XGBoost ni LSTM). Usado por
        el frontend para pre-poblar los sliders al seleccionar un departamento.

        Returns:
            dict con 'features' (dict feature→valor) y 'percentiles_locales'.
        """
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()

        df_dept = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)

        if df_dept.empty:
            raise ValueError(f"No se encontraron registros para {adm_1_name} ({iso_a0})")

        target_row = pd.DataFrame()
        if ano is not None and mes is not None:
            target_row = df_dept[(df_dept['ano'] == ano) & (df_dept['mes'] == mes)]
        if target_row.empty:
            target_row = df_dept.iloc[[-1]]

        base_record = target_row.iloc[0].to_dict()
        ref_idx = list(target_row.index)[0]
        ref_mes = int(base_record.get('mes', 1))

        cols_feat = self.agente_ml.cols_feat
        _ALWAYS_COMPUTED = {"mes_sin", "mes_cos", "incidencia_roll3", "incidencia_roll6"}
        vector = []
        for feat in cols_feat:
            val = None if feat in _ALWAYS_COMPUTED else base_record.get(feat)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                vector.append(val)
            elif "_lag" in feat:
                parts = feat.split("_lag")
                var_base = parts[0]
                lag_num = int(parts[1])
                lag_val = None
                if ref_idx >= lag_num:
                    if var_base == "incidencia_vecinos":
                        lag_row = df_dept.iloc[ref_idx - lag_num]
                        lag_ano = int(lag_row['ano'])
                        lag_mes = int(lag_row['mes'])
                        nbrs = self._neighbor_map.get((iso_a0, adm_1_name_u), [])
                        vals = [self._inc_lookup.get((iso_a0, n, lag_ano, lag_mes))
                                for n in nbrs]
                        vals = [v for v in vals if v is not None and not np.isnan(v)]
                        if vals:
                            lag_val = float(np.mean(vals))
                        else:
                            lag_val = self._inc_lookup.get(
                                (iso_a0, adm_1_name_u, lag_ano, lag_mes))
                    else:
                        map_vars = {
                            "tmax": "tmax_promedio", "tmin": "tmin_promedio",
                            "precipitacion": "precipitacion", "humedad": "humedad_promedio",
                            "incidencia": "incidencia_dengue",
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

        features = {f: float(v) for f, v in zip(cols_feat, vector)}

        p25, p50, p90 = self.p25, self.p50, self.p90
        df_d = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) &
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ]
        if not df_d.empty:
            p25 = float(df_d["incidencia_dengue"].quantile(0.25))
            p50 = max(float(df_d["incidencia_dengue"].quantile(0.50)), 0.5)
            p90 = max(float(df_d["incidencia_dengue"].quantile(0.90)), 5.0)

        return {
            "features": features,
            "percentiles_locales": {
                "p25": round(p25, 4),
                "p50": round(p50, 4),
                "p90": round(p90, 4),
            },
        }

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

    def obtener_resumen_mapa(self):
        """
        Devuelve incidencia histórica media y nivel de riesgo por departamento para el mapa.
        Los percentiles se calculan sobre las medias departamentales (no sobre registros
        mensuales), para reflejar la distribución real de riesgo entre los 187 departamentos.
        """
        grp = (
            self.df_master
            .groupby(['iso_a0', 'adm_1_name'], as_index=False)['incidencia_dengue']
            .mean()
        )

        # Percentiles sobre medias departamentales → comparación continental justa
        # Se redondean a 1 decimal para evitar que valores ~0 queden mal clasificados
        means = grp['incidencia_dengue']
        p25 = round(float(means.quantile(0.25)), 1)
        p50 = round(float(means.quantile(0.50)), 1)
        p90 = round(float(means.quantile(0.90)), 1)

        result = []
        for row in grp.itertuples():
            m = round(float(row.incidencia_dengue), 1)
            if m <= p25:
                nivel, color = "Bajo",      "#10b981"
            elif m <= p50:
                nivel, color = "Moderado",  "#eab308"
            elif m <= p90:
                nivel, color = "Alto",      "#f97316"
            else:
                nivel, color = "Muy Alto",  "#ef4444"

            result.append({
                "iso_a0":          row.iso_a0,
                "adm_1_name":      row.adm_1_name,
                "mean_incidencia": m,
                "nivel":           nivel,
                "color":           color,
            })
        return result
