# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning
--------------------------------------------------
Responsabilidad: Modelamiento temporal y espacial mediante una arquitectura
de Red Neuronal Multi-Layer Perceptron (MLP) en PyTorch.
Procesa el vector de características enriquecido en memoria.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar consola en UTF-8
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Asegurar reproducibilidad en PyTorch
torch.manual_seed(42)
np.random.seed(42)

class DengueMLPModel(nn.Module):
    """
    Arquitectura de Red Neuronal Multi-Layer Perceptron (MLP) para regresión.
    """
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

class AgentePrediccionDL:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir
            
        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.dataset_path = os.path.join(self.db_dir, "datos_procesados", "dataset_maestro_mensual_latam.csv")
        self.semilla = 42

    def generar_lags_y_vecinos_dinamico(self, df):
        """
        Calcula dinámicamente en memoria los rezagos temporales y espaciales (vecinos).
        """
        print("   [Memoria] Calculando rezagos temporales y espaciales (Opción B)...")
        df = df.copy()
        
        # 1. Rezagos temporales (lags 1, 2, 3)
        # Asegurar orden cronológico para el shift
        df = df.sort_values(by=['iso_a0', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        group = df.groupby(['iso_a0', 'adm_1_name'])
        
        # Lags climáticos
        cols_clima = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
        for var in cols_clima:
            base_name = var.split('_')[0] if 'promedio' in var else var
            for lag in [1, 2, 3]:
                df[f"{base_name}_lag{lag}"] = group[var].shift(lag)
                
        # Lags autorregresivos
        for lag in [1, 2, 3]:
            df[f"incidencia_lag{lag}"] = group['incidencia_dengue'].shift(lag)
            
        # 2. Rezagos espaciales (Vecinos)
        coords_path = os.path.join(self.db_dir, "datos_crudos", "departamentos_coordenadas.csv")
        if os.path.exists(coords_path):
            df_coords = pd.read_csv(coords_path)
            df_coords['iso_a0'] = df_coords['iso_a0'].astype(str).str.strip().str.upper()
            df_coords['adm_1_name'] = df_coords['adm_1_name'].astype(str).str.strip().str.upper()
            
            # Hacer nombres de departamento en df en mayúscula para la correspondencia
            df['adm_1_name_upper'] = df['adm_1_name'].astype(str).str.strip().str.upper()
            
            # Diccionario de coordenadas
            coords_dict = {(r.iso_a0, r.adm_1_name): (r.lat, r.lon) for r in df_coords.itertuples()}
            
            # Encontrar vecinos más cercanos (K=3) para cada departamento en su respectivo país
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
            
            # Pivot table para búsquedas rápidas de incidencia
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
            
            # Lags de incidencia de vecinos
            group_upper = df.groupby(['iso_a0', 'adm_1_name_upper'])
            for lag in [1, 2, 3]:
                df[f'incidencia_vecinos_lag{lag}'] = group_upper['incidencia_vecinos'].shift(lag)
                
            df.drop(columns=['adm_1_name_upper', 'incidencia_vecinos'], inplace=True)
            
        else:
            print("   [Advertencia] No se encontró el archivo de coordenadas. Omitiendo vecinos.")
            # Si no hay coordenadas, creamos las columnas con 0.0 para no romper la firma
            for lag in [1, 2, 3]:
                df[f'incidencia_vecinos_lag{lag}'] = 0.0
                
        # 3. Eliminar filas con nulos introducidos por los lags
        cols_lags = [c for c in df.columns if 'lag' in c]
        df.dropna(subset=cols_lags, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def entrenar_modelo_completo(self, X_train, y_train, epochs=100, lr=0.005, batch_size=256):
        """
        Entrena el modelo MLP de PyTorch utilizando mini-batches (DataLoader).
        """
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        model = DengueMLPModel(input_dim=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
            
        return model

    def entrenar_modelo(self):
        """
        Carga el dataset mensual, realiza la partición cronológica (2014-2020 / 2021-2022),
        calcula rezagos dinámicamente y entrena el modelo MLP en PyTorch.
        """
        print("[Agente 4] Cargando dataset maestro mensual consolidado...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Error: No se encontró el dataset maestro '{self.dataset_path}'.")
            
        df_raw = pd.read_csv(self.dataset_path)
        
        # Calcular lags y vecinos dinámicamente (Opción B)
        df = self.generar_lags_y_vecinos_dinamico(df_raw)
        
        # Filtrar hasta 2022 máximo
        print("   [DL/MLP] Filtrando dataset hasta el año 2022...")
        df = df[df['ano'] <= 2022].reset_index(drop=True)
        
        # Filtrado dinámico de años activos (vigilancia activa: >100 casos totales por país-año)
        print("   [DL/MLP] Aplicando filtrado dinámico de años activos (>100 casos país-año)...")
        yearly_totals = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
        df = df[yearly_totals > 100].reset_index(drop=True)
        
        # 1. Definir exclusiones y variables predictoras
        COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        
        # 2. Partición Cronológica
        print("   [DL/MLP] Particionando datos: Entrenamiento (2014-2020) | Prueba (2021-2022)")
        df_train_raw = df[df['ano'] <= 2020].copy()
        df_test_raw = df[(df['ano'] >= 2021) & (df['ano'] <= 2022)].copy()
        
        X_train_raw = df_train_raw[COLS_FEAT]
        y_train = df_train_raw['incidencia_dengue']
        X_test_raw = df_test_raw[COLS_FEAT]
        y_test = df_test_raw['incidencia_dengue']
        
        # 3. Preprocesamiento (Imputación y escalado de variables)
        imputador = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputador.fit_transform(X_train_raw), columns=COLS_FEAT)
        X_test_imp = pd.DataFrame(imputador.transform(X_test_raw), columns=COLS_FEAT)
        
        escalador = StandardScaler()
        X_train_esc = pd.DataFrame(escalador.fit_transform(X_train_imp), columns=COLS_FEAT)
        X_test_esc = pd.DataFrame(escalador.transform(X_test_imp), columns=COLS_FEAT)
        
        # 4. Validación Cruzada K-Fold (K=5) sobre el bloque de entrenamiento (2014-2020)
        print("   [DL/MLP] Iniciando validación cruzada K-Fold (K=5) en PyTorch...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.semilla)
        
        cv_mae_list, cv_rmse_list, cv_r2_list = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_esc)):
            # Splits del Fold
            X_tr, X_val = X_train_esc.iloc[train_idx], X_train_esc.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Entrenar modelo en el Fold (25 epochs para CV)
            m_fold = self.entrenar_modelo_completo(X_tr, y_tr, epochs=25, lr=0.005, batch_size=256)
            
            # Evaluar en Validación
            m_fold.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                preds_val = m_fold(val_tensor).numpy().flatten()
                preds_val = np.clip(preds_val, 0.0, None)
                
            cv_mae_list.append(mean_absolute_error(y_val, preds_val))
            cv_rmse_list.append(np.sqrt(mean_squared_error(y_val, preds_val)))
            cv_r2_list.append(r2_score(y_val, preds_val))
            
        cv_mae = np.mean(cv_mae_list)
        cv_rmse = np.mean(cv_rmse_list)
        cv_r2 = np.mean(cv_r2_list)
        print(f"   [DL/MLP] Resultados CV (Train): MAE: {cv_mae:.4f} | RMSE: {cv_rmse:.4f} | R²: {cv_r2*100:.2f}%")
        
        # 5. Entrenamiento final sobre el bloque de Entrenamiento completo (2014-2020)
        print("   [DL/MLP] Entrenando modelo final MLP en todo el Train Set...")
        modelo_mlp = self.entrenar_modelo_completo(X_train_esc, y_train, epochs=100, lr=0.005, batch_size=256)
        
        # 6. Proyección y Evaluación sobre el Conjunto de Prueba Independiente (2021-2022)
        print("   [DL/MLP] Evaluando MLP sobre Test Set (2021-2022)...")
        modelo_mlp.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test_esc.values, dtype=torch.float32)
            y_pred = modelo_mlp(test_tensor).numpy().flatten()
            y_pred = np.clip(y_pred, 0.0, None)
            
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        print(f"   [DL/MLP] Resultados Test (21-22): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2*100:.2f}%")
        
        print("SUCCESS: [Agente 4] Inferencia neuronal profunda MLP completada.")
        print("="*70)
        
        return {
            "modelo": modelo_mlp,
            "escalador": escalador,
            "imputador": imputador,
            "cols_feat": COLS_FEAT,
            "y_test": y_test,
            "X_test": X_test_raw,
            "cv_mae": cv_mae,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "y_pred": y_pred
        }

if __name__ == "__main__":
    agente = AgentePrediccionDL()
    resultados = agente.entrenar_modelo()
