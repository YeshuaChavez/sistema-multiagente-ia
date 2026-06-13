# -*- coding: utf-8 -*-
"""
Script de Entrenamiento y Serialización de Modelos
--------------------------------------------------
Entrena los modelos predictivos finalizados del Dúo Moderno (XGBoost y PyTorch MLP) 
en el bloque histórico (2014-2020) y los guarda en 'Base de Datos/modelos/' junto 
con sus escaladores, imputadores y explicaciones SHAP globales.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Asegurar codificación utf-8
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Definir la arquitectura de la red neuronal MLP
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

def generar_lags_y_vecinos_dinamico(df, db_dir):
    print("   [Procesamiento] Generando lags y vecinos en memoria...")
    df = df.copy()
    
    # 1. Lags temporales (1, 2, 3)
    df = df.sort_values(by=['iso_a0', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
    group = df.groupby(['iso_a0', 'adm_1_name'])
    
    cols_clima = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
    for var in cols_clima:
        base_name = var.split('_')[0] if 'promedio' in var else var
        for lag in [1, 2, 3]:
            df[f"{base_name}_lag{lag}"] = group[var].shift(lag)
            
    for lag in [1, 2, 3]:
        df[f"incidencia_lag{lag}"] = group['incidencia_dengue'].shift(lag)
        
    # 2. Vecinos espaciales
    coords_path = os.path.join(db_dir, "datos_crudos", "departamentos_coordenadas.csv")
    if os.path.exists(coords_path):
        df_coords = pd.read_csv(coords_path)
        df_coords['iso_a0'] = df_coords['iso_a0'].astype(str).str.strip().str.upper()
        df_coords['adm_1_name'] = df_coords['adm_1_name'].astype(str).str.strip().str.upper()
        
        df['adm_1_name_upper'] = df['adm_1_name'].astype(str).str.strip().str.upper()
        
        neighbors_dict = {}
        countries = df_coords['iso_a0'].unique()
        
        for country in countries:
            country_coords = df_coords[df_coords['iso_a0'] == country].copy()
            depts = country_coords['adm_1_name'].values
            coords_vals = country_coords[['lat', 'lon']].values
            N = len(depts)
            
            for i in range(N):
                dept_i = depts[i]
                lat_i, lon_i = coords_vals[i]
                
                distances = []
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((lat_i - coords_vals[j][0])**2 + (lon_i - coords_vals[j][1])**2)
                    distances.append((depts[j], dist))
                    
                distances.sort(key=lambda x: x[1])
                K = min(3, len(distances))
                nearest = [d[0] for d in distances[:K]]
                neighbors_dict[(country, dept_i)] = nearest
        
        lookup = {(r.iso_a0, r.adm_1_name_upper, r.ano, r.mes): r.incidencia_dengue for r in df.itertuples()}
        
        neighbor_inc = []
        for row in df.itertuples():
            key = (row.iso_a0, row.adm_1_name_upper)
            neighbors = neighbors_dict.get(key, [])
            
            if not neighbors:
                neighbor_inc.append(row.incidencia_dengue)
                continue
                
            vals = []
            for n in neighbors:
                val = lookup.get((row.iso_a0, n, row.ano, row.mes), None)
                if val is not None:
                    vals.append(val)
                    
            if vals:
                neighbor_inc.append(np.mean(vals))
            else:
                neighbor_inc.append(row.incidencia_dengue)
                
        df['incidencia_vecinos'] = neighbor_inc
        
        group_upper = df.groupby(['iso_a0', 'adm_1_name_upper'])
        for lag in [1, 2, 3]:
            df[f'incidencia_vecinos_lag{lag}'] = group_upper['incidencia_vecinos'].shift(lag)
            
        df.drop(columns=['adm_1_name_upper', 'incidencia_vecinos'], inplace=True)
    else:
        print("   [Advertencia] No se encontró el archivo de coordenadas. Lags de vecinos con 0.0.")
        for lag in [1, 2, 3]:
            df[f'incidencia_vecinos_lag{lag}'] = 0.0
            
    # Eliminar nulos de lags
    cols_lags = [c for c in df.columns if 'lag' in c]
    df.dropna(subset=cols_lags, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
    db_dir = os.path.join(base_dir, "Base de Datos")
    model_dir = os.path.join(db_dir, "modelos")
    os.makedirs(model_dir, exist_ok=True)
    
    dataset_path = os.path.join(db_dir, "datos_procesados", "dataset_maestro_mensual_latam.csv")
    print(f"Cargando dataset maestro desde: {dataset_path}")
    df_raw = pd.read_csv(dataset_path)
    
    # Calcular rezagos y vecinos
    df = generar_lags_y_vecinos_dinamico(df_raw, db_dir)
    
    # Aplicar filtrado dinámico de años activos (>100 casos totales por país-año)
    yearly_totals = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
    df = df[yearly_totals > 100].reset_index(drop=True)
    
    # Definir variables predictoras
    COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
    COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
    print(f"Características predictoras ({len(COLS_FEAT)}): {COLS_FEAT}")
    
    # Guardar las características en orden
    with open(os.path.join(model_dir, "cols_feat.pkl"), "wb") as f:
        pickle.dump(COLS_FEAT, f)
        
    # Partición Cronológica
    df_train = df[df['ano'] <= 2020].copy()
    df_test = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)].copy()
    
    X_train_raw = df_train[COLS_FEAT]
    y_train = df_train['incidencia_dengue']
    X_test_raw = df_test[COLS_FEAT]
    y_test = df_test['incidencia_dengue']
    
    # ----------------- XGBoost (ML) -----------------
    print("Entrenando modelo final XGBoost...")
    imputador_ml = SimpleImputer(strategy="median")
    X_train_imp_ml = pd.DataFrame(imputador_ml.fit_transform(X_train_raw), columns=COLS_FEAT)
    X_test_imp_ml = pd.DataFrame(imputador_ml.transform(X_test_raw), columns=COLS_FEAT)
    
    escalador_ml = StandardScaler()
    X_train_esc_ml = pd.DataFrame(escalador_ml.fit_transform(X_train_imp_ml), columns=COLS_FEAT)
    X_test_esc_ml = pd.DataFrame(escalador_ml.transform(X_test_imp_ml), columns=COLS_FEAT)
    
    # Log-transform en target
    y_train_log = np.log1p(y_train)
    
    modelo_ml = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    modelo_ml.fit(X_train_esc_ml, y_train_log)
    
    # Guardar XGBoost y preprocesadores de ML
    modelo_ml.save_model(os.path.join(model_dir, "xgboost_model.json"))
    with open(os.path.join(model_dir, "imputador_ml.pkl"), "wb") as f:
        pickle.dump(imputador_ml, f)
    with open(os.path.join(model_dir, "escalador_ml.pkl"), "wb") as f:
        pickle.dump(escalador_ml, f)
        
    print("XGBoost guardado con éxito.")
    
    # Calcular explicaciones SHAP globales precomputadas
    print("Precomputando explicabilidad SHAP (TreeSHAP)...")
    explainer = shap.TreeExplainer(modelo_ml)
    shap_vals = explainer.shap_values(X_test_esc_ml)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_importance = {var: float(val) for var, val in zip(COLS_FEAT, mean_abs_shap)}
    # Ordenar de mayor a menor
    shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(model_dir, "shap_importance.json"), "w") as f:
        json.dump(shap_importance, f, indent=4)
    print("Valores SHAP globales guardados.")

    # ----------------- PyTorch MLP (DL) -----------------
    print("Entrenando modelo final PyTorch MLP...")
    imputador_dl = SimpleImputer(strategy="median")
    X_train_imp_dl = pd.DataFrame(imputador_dl.fit_transform(X_train_raw), columns=COLS_FEAT)
    X_test_imp_dl = pd.DataFrame(imputador_dl.transform(X_test_raw), columns=COLS_FEAT)
    
    escalador_dl = StandardScaler()
    X_train_esc_dl = pd.DataFrame(escalador_dl.fit_transform(X_train_imp_dl), columns=COLS_FEAT)
    X_test_esc_dl = pd.DataFrame(escalador_dl.transform(X_test_imp_dl), columns=COLS_FEAT)
    
    # Dataset PyTorch — log1p en el target para alinear escala con XGBoost
    y_train_log_dl = np.log1p(y_train.values)
    X_train_t = torch.tensor(X_train_esc_dl.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_log_dl, dtype=torch.float32).unsqueeze(1)
    
    model_dl = DengueMLPModel(input_dim=len(COLS_FEAT))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.005, weight_decay=1e-4)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model_dl.train()
    for epoch in range(100):
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model_dl(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
    # Guardar modelo de PyTorch y preprocesadores de DL
    torch.save(model_dl.state_dict(), os.path.join(model_dir, "mlp_model.pth"))
    with open(os.path.join(model_dir, "imputador_dl.pkl"), "wb") as f:
        pickle.dump(imputador_dl, f)
    with open(os.path.join(model_dir, "escalador_dl.pkl"), "wb") as f:
        pickle.dump(escalador_dl, f)
        
    print("PyTorch MLP guardado con éxito.")
    print("Restricción e Inferencia listas. Todos los pesos han sido serializados en Base de Datos/modelos/.")
    print("=" * 80)

if __name__ == "__main__":
    main()
