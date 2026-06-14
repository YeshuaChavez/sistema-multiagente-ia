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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

        self.db_dir      = os.path.join(self.base_dir, "Base de Datos")
        self.model_dir   = os.path.join(self.db_dir, "modelos")
        self.feat_path   = os.path.join(self.db_dir, "datos_procesados", "dataset_features_latam.csv")
        self.semilla     = 42

    # ─────────────────────────────────────────────────────────────
    # MODO ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────

    def entrenar_modelo(self):
        """
        Descarga dataset_features_latam.csv de S3, entrena XGBoost con
        transformación log1p, calcula SHAP con media con signo y sube
        todos los artefactos a S3/modelos/.
        """
        print("=" * 70)
        print("  ENTRENANDO — AGENTE 3: XGBoost")
        print("=" * 70)

        os.makedirs(self.model_dir, exist_ok=True)

        # Descargar dataset de features desde S3 si no existe localmente
        s3.ensure_local(s3.PREFIX_PROCESADOS + "dataset_features_latam.csv", self.feat_path)
        if not os.path.exists(self.feat_path):
            raise FileNotFoundError(f"No se encontró el dataset de features: {self.feat_path}")

        df = pd.read_csv(self.feat_path)

        # Filtrado dinámico: solo años con >100 casos por país
        yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly > 100].reset_index(drop=True)

        COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
                        'casos_dengue', 'poblacion', 'incidencia_dengue']
        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        print(f"   [ML] Features ({len(COLS_FEAT)}): {COLS_FEAT}")

        df_train = df[df['ano'] <= 2020].copy()
        df_test  = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)].copy()

        X_train_raw = df_train[COLS_FEAT]
        y_train     = df_train['incidencia_dengue']
        X_test_raw  = df_test[COLS_FEAT]
        y_test      = df_test['incidencia_dengue']

        print(f"   [ML] Train: {len(df_train)} | Test: {len(df_test)}")

        # Imputación y escalado (fit solo en train)
        imputador = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputador.fit_transform(X_train_raw), columns=COLS_FEAT)
        X_test_imp  = pd.DataFrame(imputador.transform(X_test_raw),      columns=COLS_FEAT)

        escalador = StandardScaler()
        X_train = pd.DataFrame(escalador.fit_transform(X_train_imp), columns=COLS_FEAT)
        X_test  = pd.DataFrame(escalador.transform(X_test_imp),      columns=COLS_FEAT)

        # Transformación logarítmica del target (igual que en producción)
        y_train_log = np.log1p(y_train)

        # Entrenamiento XGBoost
        print("   [ML] Entrenando XGBoost...")
        modelo = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            random_state=self.semilla,
            n_jobs=-1,
            verbosity=0
        )
        modelo.fit(X_train, y_train_log)

        # Evaluación en test (deshacer log)
        pred_log = modelo.predict(X_test)
        pred     = np.expm1(pred_log)
        r2  = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"   [XGBoost] R²={r2*100:.2f}%  MAE={mae:.4f}")

        # SHAP con media con signo (preserva dirección del efecto)
        print("   [SHAP] Calculando TreeSHAP...")
        explainer  = shap.TreeExplainer(modelo)
        shap_vals  = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_shap  = shap_vals.mean(axis=0)
        shap_dict  = dict(sorted(
            {f: float(v) for f, v in zip(COLS_FEAT, mean_shap)}.items(),
            key=lambda x: abs(x[1]), reverse=True
        ))

        # Serializar artefactos localmente
        with open(os.path.join(self.model_dir, "xgb_model.pkl"),    "wb") as f: pickle.dump(modelo,     f)
        with open(os.path.join(self.model_dir, "imputador_ml.pkl"),  "wb") as f: pickle.dump(imputador,  f)
        with open(os.path.join(self.model_dir, "escalador_ml.pkl"),  "wb") as f: pickle.dump(escalador,  f)
        with open(os.path.join(self.model_dir, "cols_feat.pkl"),     "wb") as f: pickle.dump(COLS_FEAT,  f)
        with open(os.path.join(self.model_dir, "shap_importance.json"), "w") as f:
            json.dump(shap_dict, f, indent=4)

        # Subir a S3
        for fname in ["xgb_model.pkl", "imputador_ml.pkl", "escalador_ml.pkl",
                      "cols_feat.pkl", "shap_importance.json"]:
            s3.upload(os.path.join(self.model_dir, fname), s3.PREFIX_MODELOS + fname)

        print("SUCCESS: [Agente 3] XGBoost entrenado y subido a S3.")
        print("=" * 70)

        return {"r2_xgb": round(r2, 4), "mae_xgb": round(mae, 4),
                "n_train": len(df_train), "n_test": len(df_test)}

    # ─────────────────────────────────────────────────────────────
    # MODO INFERENCIA
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def cargar_modelo(cls, model_dir, base_dir=None):
        """Carga XGBoost serializado para inferencia sin reentrenar."""
        agente = cls(base_dir=base_dir)
        with open(os.path.join(model_dir, "xgb_model.pkl"),   "rb") as f: agente.modelo    = pickle.load(f)
        with open(os.path.join(model_dir, "imputador_ml.pkl"), "rb") as f: agente.imputador = pickle.load(f)
        with open(os.path.join(model_dir, "escalador_ml.pkl"), "rb") as f: agente.escalador = pickle.load(f)
        with open(os.path.join(model_dir, "cols_feat.pkl"),    "rb") as f: agente.cols_feat = pickle.load(f)

        shap_path = os.path.join(model_dir, "shap_importance.json")
        agente.shap_importance = json.load(open(shap_path)) if os.path.exists(shap_path) else {}
        agente._shap_explainer = shap.TreeExplainer(agente.modelo)
        print(f"   [Agente 3] XGBoost cargado — {len(agente.cols_feat)} features.")
        return agente

    def predecir(self, vector, compute_shap=False):
        entrada  = pd.DataFrame([vector], columns=self.cols_feat)
        X_imp    = self.imputador.transform(entrada)
        X_esc    = pd.DataFrame(self.escalador.transform(X_imp), columns=self.cols_feat)
        pred_log = float(self.modelo.predict(X_esc)[0])
        pred     = max(0.0, np.expm1(pred_log))
        result   = {"prediccion_ml": round(pred, 4)}

        if compute_shap and hasattr(self, '_shap_explainer') and self._shap_explainer is not None:
            shap_vals = self._shap_explainer.shap_values(X_esc)
            if hasattr(shap_vals, 'values'):
                raw = shap_vals.values[0]
            elif isinstance(shap_vals, list):
                raw = shap_vals[0][0]
            else:
                raw = np.asarray(shap_vals)[0]
            result["shap_local"] = {f: round(float(v), 6) for f, v in zip(self.cols_feat, raw)}
        return result
