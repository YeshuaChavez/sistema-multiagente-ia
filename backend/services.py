# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Backend Services
----------------------------
Carga persistente de datos y modelos predictivos en memoria.
Realiza imputación, escalado, simulaciones y cálculo de Ensemble y Riesgo.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBRegressor

# Definir la arquitectura de la red neuronal MLP (debe coincidir con la de entrenamiento)
class DengueMLPModel(nn.Module):
    def __init__(self, input_dim=23, output_dim=1):
        super(DengueMLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

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
        self.modelo_ml = None
        self.modelo_dl = None
        
        self.imputador_ml = None
        self.escalador_ml = None
        self.imputador_dl = None
        self.escalador_dl = None
        self.shap_importance = None
        
        self.p25 = 0.0
        self.p50 = 0.0
        self.p90 = 0.0
        
        self.inicializar_servicio()

    def inicializar_servicio(self):
        print("[In-Memory Service] Iniciando carga de activos en RAM...")
        
        # 1. Cargar Dataset Maestro
        master_path = os.path.join(self.processed_dir, "dataset_maestro_mensual_latam.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Falta dataset maestro mensual en: {master_path}")
        self.df_master = pd.read_csv(master_path)
        print(f"   -> Dataset cargado: {self.df_master.shape[0]} registros.")
        
        # Calcular percentiles globales de incidencia para niveles de riesgo
        self.p25 = float(self.df_master["incidencia_dengue"].quantile(0.25))
        self.p50 = float(self.df_master["incidencia_dengue"].quantile(0.50))
        self.p90 = float(self.df_master["incidencia_dengue"].quantile(0.90))
        
        # 2. Cargar Coordenadas
        coords_path = os.path.join(self.raw_dir, "departamentos_coordenadas.csv")
        if os.path.exists(coords_path):
            self.df_coords = pd.read_csv(coords_path)
            self.df_coords['iso_a0'] = self.df_coords['iso_a0'].astype(str).str.strip().str.upper()
            self.df_coords['adm_1_name'] = self.df_coords['adm_1_name'].astype(str).str.strip().str.upper()
            
        # 3. Cargar Nombres de Variables
        cols_feat_path = os.path.join(self.model_dir, "cols_feat.pkl")
        if not os.path.exists(cols_feat_path):
            raise FileNotFoundError(f"Falta lista de variables en: {cols_feat_path}")
        with open(cols_feat_path, "rb") as f:
            self.cols_feat = pickle.load(f)
            
        # 4. Cargar XGBoost (ML)
        xgb_path = os.path.join(self.model_dir, "xgboost_model.json")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"Falta modelo XGBoost en: {xgb_path}")
        self.modelo_ml = XGBRegressor()
        self.modelo_ml.load_model(xgb_path)
        
        with open(os.path.join(self.model_dir, "imputador_ml.pkl"), "rb") as f:
            self.imputador_ml = pickle.load(f)
        with open(os.path.join(self.model_dir, "escalador_ml.pkl"), "rb") as f:
            self.escalador_ml = pickle.load(f)
            
        # 5. Cargar PyTorch MLP (DL)
        mlp_path = os.path.join(self.model_dir, "mlp_model.pth")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"Falta pesos MLP en: {mlp_path}")
        self.modelo_dl = DengueMLPModel(input_dim=len(self.cols_feat))
        self.modelo_dl.load_state_dict(torch.load(mlp_path, map_location=torch.device('cpu')))
        self.modelo_dl.eval()
        
        with open(os.path.join(self.model_dir, "imputador_dl.pkl"), "rb") as f:
            self.imputador_dl = pickle.load(f)
        with open(os.path.join(self.model_dir, "escalador_dl.pkl"), "rb") as f:
            self.escalador_dl = pickle.load(f)
            
        # 6. Cargar SHAP Global
        shap_path = os.path.join(self.model_dir, "shap_importance.json")
        if os.path.exists(shap_path):
            with open(shap_path, "r") as f:
                self.shap_importance = json.load(f)
                
        print("SUCCESS: [In-Memory Service] Todos los modelos y datos cargados en la memoria RAM.")

    def calcular_nivel_riesgo(self, pred_val):
        if pred_val <= self.p25:
            return {"nivel": "Bajo / Normal", "codigo": "normal", "color": "#10b981"}
        elif pred_val <= self.p50:
            return {"nivel": "Vigilancia", "codigo": "vigilancia", "color": "#eab308"}
        elif pred_val <= self.p90:
            return {"nivel": "Alerta", "codigo": "alerta", "color": "#f97316"}
        else:
            return {"nivel": "Epidemia", "codigo": "epidemia", "color": "#ef4444"}

    def realizar_prediccion_vector(self, vector_x):
        """
        Realiza la predicción basándose en un vector de entrada ordenado (23 features).
        """
        entrada = pd.DataFrame([vector_x], columns=self.cols_feat)
        
        # 1. Inferencia XGBoost (ML)
        entrada_imp_ml = self.imputador_ml.transform(entrada)
        entrada_esc_ml = self.escalador_ml.transform(entrada_imp_ml)
        pred_ml_log = float(self.modelo_ml.predict(entrada_esc_ml)[0])
        pred_ml = max(0.0, np.expm1(pred_ml_log))
        
        # 2. Inferencia PyTorch MLP (DL)
        entrada_imp_dl = self.imputador_dl.transform(entrada)
        entrada_esc_dl = self.escalador_dl.transform(entrada_imp_dl)
        with torch.no_grad():
            x_tensor = torch.tensor(entrada_esc_dl, dtype=torch.float32)
            pred_dl = float(self.modelo_dl(x_tensor).numpy()[0][0])
        pred_dl = max(0.0, pred_dl)
        
        # 3. Fusión Ensemble Promedio
        pred_ens = (pred_ml + pred_dl) / 2.0
        
        # Clasificar riesgos
        riesgo_ml = self.calcular_nivel_riesgo(pred_ml)
        riesgo_dl = self.calcular_nivel_riesgo(pred_dl)
        riesgo_ens = self.calcular_nivel_riesgo(pred_ens)
        
        return {
            "prediccion_ml": round(pred_ml, 4),
            "riesgo_ml": riesgo_ml,
            "prediccion_dl": round(pred_dl, 4),
            "riesgo_dl": riesgo_dl,
            "prediccion_ensemble": round(pred_ens, 4),
            "riesgo_ensemble": riesgo_ens
        }

    def simular_prediccion_departamento(self, iso_a0, adm_1_name, ano, mes, clima_overrides=None):
        """
        Busca el registro departamental y construye dinámicamente los rezagos y vecinos.
        """
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()
        
        # Filtrar el registro del departamento para el año y mes seleccionado
        df_dept = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) & 
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)
        
        if df_dept.empty:
            raise ValueError(f"No se encontraron registros históricos para {adm_1_name} ({iso_a0})")
            
        # Buscar el registro objetivo
        target_row = df_dept[(df_dept['ano'] == ano) & (df_dept['mes'] == mes)]
        
        if target_row.empty:
            # Si no hay registro del mes exacto, usar la mediana histórica de ese departamento
            mediana_dept = df_dept.median(numeric_only=True).to_dict()
            base_record = {col: mediana_dept.get(col, 0.0) for col in self.df_master.columns if col not in ['iso_a0', 'pais', 'adm_1_name']}
            base_record['iso_a0'] = iso_a0
            base_record['adm_1_name'] = adm_1_name
            base_record['ano'] = ano
            base_record['mes'] = mes
        else:
            base_record = target_row.iloc[0].to_dict()
            
        # Aplicar modificaciones del clima del usuario
        if clima_overrides:
            for key, val in clima_overrides.items():
                if key in base_record:
                    base_record[key] = float(val)
                    
        # Construir el vector de 23 características (en el orden exacto de cols_feat)
        # Para simplificar la inferencia en vivo:
        # Extraemos los valores de lags de la serie temporal del departamento (si existen)
        idx_target = df_dept[(df_dept['ano'] == ano) & (df_dept['mes'] == mes)].index
        
        vector = []
        for feat in self.cols_feat:
            # 1. Si es variable actual
            if feat in base_record:
                vector.append(base_record[feat])
            # 2. Si es un rezago temporal (lag1, lag2, lag3)
            elif "_lag" in feat:
                parts = feat.split("_lag")
                var_base = parts[0]
                lag_num = int(parts[1])
                
                # Intentar buscar en el histórico real
                val = None
                if len(idx_target) > 0:
                    idx = idx_target[0]
                    if idx >= lag_num:
                        # Mapeo de nombres climáticos simplificados
                        map_vars = {
                            "tmax": "tmax_promedio", "tmin": "tmin_promedio",
                            "precipitacion": "precipitacion", "humedad": "humedad_promedio",
                            "incidencia": "incidencia_dengue", "incidencia_vecinos": "incidencia_dengue" # Simplificado
                        }
                        col_real = map_vars.get(var_base, var_base)
                        if col_real in df_dept.columns:
                            val = df_dept.loc[idx - lag_num, col_real]
                            
                # Fallback a la mediana si no hay registro temporal
                if val is None or pd.isna(val):
                    # Mediana histórica de la variable
                    val_med = df_dept[df_dept['mes'] == mes].median(numeric_only=True).to_dict()
                    val = val_med.get(var_base, 0.0)
                    
                vector.append(val)
            else:
                vector.append(0.0)
                
        # Aplicar modificaciones en base al nombre exacto de la variable predictora final (lags, agua, clima, etc.)
        if clima_overrides:
            for i, feat in enumerate(self.cols_feat):
                if feat in clima_overrides:
                    vector[i] = float(clima_overrides[feat])
                
        # Realizar predicción
        res = self.realizar_prediccion_vector(vector)
        res["features_usadas"] = {feat: float(val) for feat, val in zip(self.cols_feat, vector)}
        return res

    def obtener_metadatos_paises(self):
        """
        Retorna la estructura geográfica y los años de datos.
        """
        paises_dict = {}
        for row in self.df_master[['pais', 'iso_a0', 'adm_1_name']].drop_duplicates().itertuples():
            if row.pais not in paises_dict:
                paises_dict[row.pais] = {
                    "iso_a0": row.iso_a0,
                    "departamentos": []
                }
            if row.adm_1_name not in paises_dict[row.pais]["departamentos"]:
                paises_dict[row.pais]["departamentos"].append(row.adm_1_name)
                
        # Ordenar departamentos
        for p in paises_dict:
            paises_dict[p]["departamentos"].sort()
            
        return paises_dict

    def obtener_historico_departamento(self, iso_a0, adm_1_name):
        """
        Retorna la serie temporal completa del departamento para graficar en frontend.
        """
        iso_a0 = iso_a0.strip().upper()
        adm_1_name_u = adm_1_name.strip().upper()
        
        df_filtered = self.df_master[
            (self.df_master['iso_a0'] == iso_a0) & 
            (self.df_master['adm_1_name'].str.upper() == adm_1_name_u)
        ].sort_values(['ano', 'mes']).reset_index(drop=True)
        
        # Convertir a JSON records
        records = []
        for r in df_filtered.itertuples():
            records.append({
                "fecha": f"{r.ano}-{r.mes:02d}",
                "ano": int(r.ano),
                "mes": int(r.mes),
                "casos": int(r.casos_dengue),
                "incidencia": float(r.incidencia_dengue),
                "tmax": float(r.tmax_promedio),
                "tmin": float(r.tmin_promedio),
                "precipitacion": float(r.precipitacion),
                "humedad": float(r.humedad_promedio)
            })
        return records
