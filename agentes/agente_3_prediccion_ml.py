# -*- coding: utf-8 -*-
"""
UNMSM | Trabajo de Grado | SMA-ML/DL
Agente 3: Predicción Machine Learning
--------------------------------------------------
Responsabilidad: Modelamiento continuo de la incidencia de dengue mediante XGBoost,
optimización bajo validación cruzada temporal y análisis de explicabilidad (XAI)
mediante la extracción de valores SHAP (Shapley Additive exPlanations).
"""

import os
import sys
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

class AgentePrediccionML:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir
            
        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.dataset_path = os.path.join(self.db_dir, "dataset_maestro_mensual_latam.csv")
        self.semilla = 42

    def entrenar_modelo(self):
        """
        Carga el dataset maestro, realiza la partición cronológica (2014-2020 / 2021-2022)
        y entrena el regresor XGBoost con validación cruzada y explicabilidad SHAP.
        """
        print("[Agente 3] Cargando dataset maestro mensual consolidado...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Error: No se encontró el dataset maestro '{self.dataset_path}'.")
            
        df = pd.read_csv(self.dataset_path)
        
        # 1. Definir exclusiones y variables predictoras
        COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        
        # 2. Partición Cronológica
        print("   [ML] Particionando datos: Entrenamiento (2014-2020) | Prueba (2021-2022)")
        df_train_raw = df[df['ano'] <= 2020].copy()
        df_test_raw = df[df['ano'] >= 2021].copy()
        
        X_train_raw = df_train_raw[COLS_FEAT]
        y_train = df_train_raw['incidencia_dengue']
        X_test_raw = df_test_raw[COLS_FEAT]
        y_test = df_test_raw['incidencia_dengue']
        
        len_train = len(df_train_raw)
        len_test = len(df_test_raw)
        print(f"   [ML] Registros de Train: {len_train} | Test: {len_test}")
        
        # 3. Preprocesamiento (Ajustado solo con Train para evitar data leakage)
        imputador = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputador.fit_transform(X_train_raw), columns=COLS_FEAT)
        X_test_imp = pd.DataFrame(imputador.transform(X_test_raw), columns=COLS_FEAT)
        
        escalador = StandardScaler()
        X_train = pd.DataFrame(escalador.fit_transform(X_train_imp), columns=COLS_FEAT)
        X_test = pd.DataFrame(escalador.transform(X_test_imp), columns=COLS_FEAT)
        
        # 4. Sintonización y Validación Cruzada de XGBoost (K=5 Folds) en el bloque histórico (2014-2020)
        print("   [ML] Iniciando validación cruzada K-Fold (K=5) para XGBoost...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.semilla)
        modelo_xgb = XGBRegressor(random_state=self.semilla, verbosity=0, eval_metric="rmse", n_jobs=-1)
        
        cv = cross_validate(
            modelo_xgb, X_train, y_train, cv=kfold,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=-1
        )
        
        cv_mae = -cv["test_mae"].mean()
        cv_rmse = -cv["test_rmse"].mean()
        cv_r2 = cv["test_r2"].mean()
        print(f"   [ML] Resultados CV (Train): MAE: {cv_mae:.4f} | RMSE: {cv_rmse:.4f} | R²: {cv_r2*100:.2f}%")
        
        # 5. Entrenamiento final sobre el bloque de Entrenamiento completo (2014-2020)
        print("   [ML] Entrenando modelo final en todo el Train Set...")
        modelo_xgb.fit(X_train, y_train)
        
        # 6. Proyección y Evaluación sobre el Conjunto de Prueba Independiente (2021-2022)
        print("   [ML] Evaluando sobre Test Set (2021-2022)...")
        y_pred = modelo_xgb.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, None)  # La incidencia no puede ser negativa
        
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        print(f"   [ML] Resultados Test (21-22): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2*100:.2f}%")
        
        # 7. Capa de Explicabilidad (XAI) mediante Valores SHAP
        print("   [XAI/SHAP] Extrayendo valores de Shapley mediante TreeSHAP...")
        explainer = shap.TreeExplainer(modelo_xgb)
        # Calcular valores SHAP locales para el conjunto de prueba
        shap_vals = explainer.shap_values(X_test)
        
        # En algunas versiones de shap, shap_values puede ser una lista o un objeto específico.
        # Nos aseguramos de extraer el array y calcular la importancia media absoluta (global)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Generar Series de importancia ordenada
        shap_importance = pd.Series(mean_abs_shap, index=COLS_FEAT).sort_values(ascending=False).head(10)
        print("   [XAI/SHAP] Top 5 variables más influyentes:")
        for idx, (var, val) in enumerate(shap_importance.head(5).items()):
            print(f"     {idx+1}. {var}: {val:.4f} (SHAP medio)")
            
        print("SUCCESS: [Agente 3] Entrenamiento y análisis explicable (SHAP) finalizado.")
        print("="*70)
        
        return {
            "modelo": modelo_xgb,
            "escalador": escalador,
            "imputador": imputador,
            "cols_feat": COLS_FEAT,
            "y_test": y_test,
            "X_test": X_test,
            "df": df,
            "cv_mae": cv_mae,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "y_pred": y_pred,
            "shap_importance": shap_importance,
            "n_train": len_train,
            "n_test": len_test
        }

if __name__ == "__main__":
    agente = AgentePrediccionML()
    resultados = agente.entrenar_modelo()
