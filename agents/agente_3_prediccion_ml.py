# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 3: Predicción Machine Learning (XGBoost)
-------------------------------------------------
Responsabilidad: Entrenar el modelo XGBoost sobre el dataset de features
generado por el Agente 2, calcular SHAP y serializar todos los artefactos
en S3. En modo inferencia, cargar el modelo serializado para predicción online.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3


class AgentePrediccionML:
    def __init__(self, base_dir=None):
        if base_dir is None:
            if os.path.exists("/app"):
                self.base_dir = "/app"
            else:
                self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.model_dir   = os.path.join(self.base_dir, "data", "models")
        self.feat_path   = os.path.join(self.base_dir, "data", "processed", "dataset_features_latam.csv")
        self.semilla     = 42

    # ─────────────────────────────────────────────────────────────
    # MODO ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────

    def entrenar_modelo(self):
        """
        Ciclo de vida completo del modelo ML (Agente 3 — XGBoost):
          Fase 1  — Definicion del problema: prediccion de tasa de incidencia de dengue
                    a escala subnacional mensual (ver documentacion del SMA)
          Fase 2  — Recoleccion de datos: ejecutada por Agente 1 (agente_1_recoleccion.py)
          Fase 3  — Preparacion de datos: ejecutada por Agente 2 (agente_2_preprocesamiento.py)
          Fase 4  — Division del conjunto: particion cronologica dinamica (ultimos 2 anos = test,
                    resto = train); permite reentrenamiento automatico sin cambiar codigo
          Fase 5  — Seleccion del modelo: XGBRegressor dentro de Pipeline sklearn
                    (SimpleImputer + StandardScaler + XGBRegressor)
          Fase 6a — Entrenamiento baseline con parametros por defecto
          Fase 7a — Evaluacion del baseline (R2, MAE en test set)
          Fase 8  — Optimizacion de hiperparametros: Optuna TPE, 50 trials x K=5 folds
                    cronologicos; sampler TPESampler(seed=42)
          Fase 6b — Reentrenamiento con best_estimator_ (parametros optimos, refit=True)
          Fase 7b — Evaluacion final del modelo optimizado
          Fase 9  — Implementacion: serializacion y subida a AWS S3, carga en FastAPI/Railway
          Fase 10 — Mantenimiento: drift detection (PSI sobre features climaticas NASA POWER)
                    + reentrenamiento automatico via GitHub Actions cuando llega nueva version
                    OpenDengue (verificar_actualizacion.py ejecutado el 1ro de cada mes)
        """
        print("=" * 70)
        print("  ENTRENANDO — AGENTE 3: XGBoost")
        print("=" * 70)

        os.makedirs(self.model_dir, exist_ok=True)

        s3.ensure_local(s3.PREFIX_PROCESADOS + "dataset_features_latam.csv", self.feat_path)
        if not os.path.exists(self.feat_path):
            raise FileNotFoundError(f"No se encontró el dataset de features: {self.feat_path}")

        df = pd.read_csv(self.feat_path)

        yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly > 100].reset_index(drop=True)

        COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
                        'casos_dengue', 'poblacion', 'incidencia_dengue']
        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        print(f"   [ML] Features ({len(COLS_FEAT)}): {COLS_FEAT}")

        # ── Fase 4: División cronológica del conjunto (evita data leakage) ──
        # Split dinámico: últimos 2 años = test, el resto = train
        # Permite reentrenamiento automático cuando llegan nuevos datos de OpenDengue
        TEST_ANOS = 2
        max_ano   = int(df['ano'].max())
        split_ano = max_ano - TEST_ANOS        # e.g. max=2022 → split=2020; max=2023 → split=2021
        df_train = df[df['ano'] <= split_ano].copy()
        df_test  = df[df['ano'] >  split_ano].copy()
        print(f"   [ML] Split dinámico: train ≤{split_ano} | test >{split_ano} (max={max_ano})")

        X_train_raw = df_train[COLS_FEAT]
        y_train     = df_train['incidencia_dengue']
        X_test_raw  = df_test[COLS_FEAT]
        y_test      = df_test['incidencia_dengue']

        print(f"   [ML] Train: {len(df_train)} | Test: {len(df_test)}")

        y_train_log = np.log1p(y_train)

        # ── Fase 5: Selección del modelo — Pipeline sklearn (imputador + escalador + XGBoost) ──
        # ── Fase 6a: Baseline con parámetros por defecto de XGBoost ──
        print("\n   [Fase 6a] Entrenando baseline con parametros por defecto...")
        pipeline_base = Pipeline([
            ('imputador', SimpleImputer(strategy='median')),
            ('escalador', StandardScaler()),
            ('modelo',    XGBRegressor(random_state=self.semilla, n_jobs=-1, verbosity=0))
        ])
        pipeline_base.fit(X_train_raw, y_train_log)

        # ── Fase 7a: Evaluación del baseline ──
        pred_base = np.expm1(pipeline_base.predict(X_test_raw))
        r2_base   = r2_score(y_test, pred_base)
        mae_base  = mean_absolute_error(y_test, pred_base)
        print(f"   [Fase 7a] Baseline — R²={r2_base*100:.2f}%  MAE={mae_base:.4f}")

        # ── Fase 8: Bayesian Optimization (Optuna TPE) + folds cronologicos ──
        N_TRIALS_XGB   = 50
        anos_train_arr = df_train['ano'].values
        fold_val_anos  = sorted(set(anos_train_arr.tolist()))[-5:]
        folds_xgb = [
            (anos_train_arr < val_ano, anos_train_arr == val_ano)
            for val_ano in fold_val_anos
        ]
        X_tr_np      = X_train_raw.values
        y_tr_log_np  = y_train_log.values

        print(f"\n   [Fase 8] Optuna TPE — {N_TRIALS_XGB} trials x K=5 folds cronologicos...")
        print(f"   Folds val: {fold_val_anos}")

        def objective_xgb(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
                "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
                "objective": "reg:squarederror",
                "random_state": self.semilla, "verbosity": 0, "n_jobs": -1,
            }
            r2s = []
            for tr_mask, val_mask in folds_xgb:
                if val_mask.sum() == 0:
                    continue
                imp = SimpleImputer(strategy='median')
                sc  = StandardScaler()
                Xtr = sc.fit_transform(imp.fit_transform(X_tr_np[tr_mask]))
                Xvl = sc.transform(imp.transform(X_tr_np[val_mask]))
                m   = XGBRegressor(**params)
                m.fit(Xtr, y_tr_log_np[tr_mask])
                pv  = m.predict(Xvl)
                r2s.append(r2_score(y_tr_log_np[val_mask], pv))
            return float(np.mean(r2s))

        def cb_xgb(study, trial):
            best = " <-- mejor" if trial.value == study.best_value else ""
            p = trial.params
            print(f"  Trial {trial.number+1:02d}  n_est={p['n_estimators']}  "
                  f"lr={p['learning_rate']:.4f}  depth={p['max_depth']}  "
                  f"R2_CV={trial.value*100:.2f}%{best}")

        study_xgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.semilla)
        )
        study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, callbacks=[cb_xgb])
        best_trial = study_xgb.best_trial
        print(f"\n   Mejor R2 CV: {best_trial.value*100:.2f}%")
        for k, v in sorted(best_trial.params.items()):
            print(f"     {k:22s}: {v}")

        # ── Fase 6b: Reentrenar con mejores params en todo el train set ──
        best_xgb_params = {
            **best_trial.params,
            "objective": "reg:squarederror",
            "random_state": self.semilla, "verbosity": 0, "n_jobs": -1,
        }
        pipeline = Pipeline([
            ('imputador', SimpleImputer(strategy='median')),
            ('escalador', StandardScaler()),
            ('modelo',    XGBRegressor(**best_xgb_params))
        ])
        pipeline.fit(X_train_raw, y_train_log)
        pred_log = pipeline.predict(X_test_raw)
        pred     = np.expm1(pred_log)
        r2       = r2_score(y_test, pred)
        mae      = mean_absolute_error(y_test, pred)
        # R² en escala logarítmica (estándar epidemiológico para datos sesgados)
        r2_log   = r2_score(np.log1p(y_test), pred_log)
        print(f"\n   [Fase 7b] Optimizado — R²={r2_log*100:.2f}%  MAE={mae:.4f}")
        print(f"   Mejora sobre baseline: R² +{(r2-r2_base)*100:.2f}pp  MAE {mae-mae_base:+.4f}")

        # SHAP — se accede al modelo dentro del pipeline con named_steps
        print("   [SHAP] Calculando TreeSHAP...")
        modelo    = pipeline.named_steps['modelo']
        X_test_sc = pipeline[:-1].transform(X_test_raw)   # imputa + escala sin predecir
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_test_sc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_shap = shap_vals.mean(axis=0)
        shap_dict = dict(sorted(
            {f: float(v) for f, v in zip(COLS_FEAT, mean_shap)}.items(),
            key=lambda x: abs(x[1]), reverse=True
        ))

        # Serializar: pipeline completo + artefactos individuales (compatibilidad)
        with open(os.path.join(self.model_dir, "pipeline_ml.pkl"),   "wb") as f: pickle.dump(pipeline,  f)
        with open(os.path.join(self.model_dir, "xgb_model.pkl"),     "wb") as f: pickle.dump(modelo,    f)
        with open(os.path.join(self.model_dir, "imputador_ml.pkl"),  "wb") as f: pickle.dump(pipeline.named_steps['imputador'], f)
        with open(os.path.join(self.model_dir, "escalador_ml.pkl"),  "wb") as f: pickle.dump(pipeline.named_steps['escalador'], f)
        with open(os.path.join(self.model_dir, "cols_feat.pkl"),     "wb") as f: pickle.dump(COLS_FEAT, f)
        with open(os.path.join(self.model_dir, "shap_importance.json"), "w") as f:
            json.dump(shap_dict, f, indent=4)

        xgb_config = {
            "best_params":    best_trial.params,
            "r2":             round(float(r2_log), 4),
            "mae":            round(float(mae), 4),
            "r2_baseline":    round(float(r2_base), 4),
            "r2_cv_mejor":    round(float(best_trial.value), 4),
            "k_folds":        5,
            "optimizer":      "Optuna/TPE",
            "n_trials":       N_TRIALS_XGB,
            "trials": [{"trial": t.number, "params": t.params, "r2_cv": round(t.value, 4)}
                       for t in study_xgb.trials],
        }
        with open(os.path.join(self.model_dir, "xgb_config.json"), "w") as f:
            json.dump(xgb_config, f, indent=4)

        # Subir a S3
        for fname in ["pipeline_ml.pkl", "xgb_model.pkl", "imputador_ml.pkl",
                      "escalador_ml.pkl", "cols_feat.pkl", "shap_importance.json",
                      "xgb_config.json"]:
            s3.upload(os.path.join(self.model_dir, fname), s3.PREFIX_MODELOS + fname)

        print("SUCCESS: [Agente 3] XGBoost entrenado y subido a S3.")
        print("=" * 70)

        # Lookup de predicciones XGBoost en test: (iso_a0, adm_upper, ano, mes) → pred
        xgb_test_lookup = {
            (r.iso_a0.strip().upper(), r.adm_1_name.strip().upper(), int(r.ano), int(r.mes)): float(p)
            for r, p in zip(df_test.itertuples(), pred)
        }

        return {"r2_xgb": round(r2_log, 4), "mae_xgb": round(mae, 4),
                "n_train": len(df_train), "n_test": len(df_test),
                "xgb_test_lookup": xgb_test_lookup}

    # ─────────────────────────────────────────────────────────────
    # MODO INFERENCIA
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def cargar_modelo(cls, model_dir, base_dir=None):
        """Carga el Pipeline serializado para inferencia sin reentrenar."""
        agente = cls(base_dir=base_dir)

        pipeline_path = os.path.join(model_dir, "pipeline_ml.pkl")
        if os.path.exists(pipeline_path):
            # Pipeline completo: imputa + escala + predice en un solo paso
            with open(pipeline_path, "rb") as f:
                agente.pipeline = pickle.load(f)
            steps = agente.pipeline.named_steps
            # Compatible con pipelines nuevos (scaler/model) y antiguos (imputador/escalador/modelo)
            agente.modelo    = steps.get('modelo') or steps.get('model')
            agente.escalador = steps.get('escalador') or steps.get('scaler')
            agente.imputador = steps.get('imputador')
        else:
            # Compatibilidad con artefactos anteriores (sin pipeline)
            with open(os.path.join(model_dir, "xgb_model.pkl"),    "rb") as f: agente.modelo    = pickle.load(f)
            with open(os.path.join(model_dir, "imputador_ml.pkl"),  "rb") as f: agente.imputador = pickle.load(f)
            with open(os.path.join(model_dir, "escalador_ml.pkl"),  "rb") as f: agente.escalador = pickle.load(f)
            agente.pipeline = Pipeline([
                ('imputador', agente.imputador),
                ('escalador', agente.escalador),
                ('modelo',    agente.modelo),
            ])

        with open(os.path.join(model_dir, "cols_feat.pkl"), "rb") as f:
            agente.cols_feat = pickle.load(f)

        shap_path = os.path.join(model_dir, "shap_importance.json")
        agente.shap_importance = json.load(open(shap_path)) if os.path.exists(shap_path) else {}
        agente._shap_explainer = shap.TreeExplainer(agente.modelo)
        print(f"   [Agente 3] Pipeline XGBoost cargado — {len(agente.cols_feat)} features.")
        return agente

    def predecir(self, vector, compute_shap=False):
        entrada  = pd.DataFrame([vector], columns=self.cols_feat)
        # Pipeline aplica imputación + escalado + predicción en un solo paso
        pred_log = float(self.pipeline.predict(entrada)[0])
        pred     = max(0.0, np.expm1(pred_log))
        result   = {"prediccion_ml": round(pred, 4)}

        if compute_shap and hasattr(self, '_shap_explainer') and self._shap_explainer is not None:
            X_sc     = self.pipeline[:-1].transform(entrada)   # imputa + escala sin predecir
            shap_vals = self._shap_explainer.shap_values(X_sc)
            if hasattr(shap_vals, 'values'):
                raw = shap_vals.values[0]
            elif isinstance(shap_vals, list):
                raw = shap_vals[0][0]
            else:
                raw = np.asarray(shap_vals)[0]
            result["shap_local"] = {f: round(float(v), 6) for f, v in zip(self.cols_feat, raw)}
        return result
