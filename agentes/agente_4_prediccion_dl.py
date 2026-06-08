# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 4: Predicción Deep Learning
--------------------------------------------------
Responsabilidad: Modelamiento temporal de largo alcance mediante una arquitectura
de Red Neuronal Recurrente LSTM (Long Short-Term Memory) en PyTorch.
Procesa ventanas secuenciales de 3 meses para capturar dinámicas no lineales.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar reproducibilidad en PyTorch
torch.manual_seed(42)
np.random.seed(42)

class DengueLSTMModel(nn.Module):
    """
    Arquitectura de Red Recurrente LSTM con covariables estáticas/futuras.
    """
    def __init__(self, seq_features=5, hidden_dim=32, static_features=5, output_dim=1):
        super(DengueLSTMModel, self).__init__()
        # Capa LSTM para la secuencia histórica de 3 meses (lags 1, 2, 3)
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        # Capa densa conectada que combina el estado de la LSTM y las variables estáticas/actuales
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + static_features, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, seq_x, static_x):
        # seq_x: (batch_size, seq_len=3, seq_features=5)
        # static_x: (batch_size, static_features=5)
        out, (hn, cn) = self.lstm(seq_x)
        # Extraer el estado oculto del último paso temporal (t=3)
        last_hidden = out[:, -1, :]
        # Concatenar con variables estáticas/actuales
        x_concat = torch.cat([last_hidden, static_x], dim=1)
        # Predicción final
        pred = self.fc(x_concat)
        return pred

class AgentePrediccionDL:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir
            
        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.dataset_path = os.path.join(self.db_dir, "dataset_maestro_mensual_latam.csv")
        self.semilla = 42

    def preparar_secuencias(self, X_df):
        """
        Divide la matriz X de 20 features en:
        - Una secuencia temporal de 3 meses: seq_x (shape: N, 3, 5)
        - Un vector de covariables actuales/agua: static_x (shape: N, 5)
        """
        N = len(X_df)
        
        # 1. Reconstruir secuencias (lags 3, 2, 1) para las 5 variables principales
        # Variables: tmax, tmin, precipitacion, humedad, incidencia
        seq_x = np.zeros((N, 3, 5))
        
        # Paso 1: Hace 3 meses (Lag 3)
        seq_x[:, 0, 0] = X_df['tmax_lag3']
        seq_x[:, 0, 1] = X_df['tmin_lag3']
        seq_x[:, 0, 2] = X_df['precipitacion_lag3']
        seq_x[:, 0, 3] = X_df['humedad_lag3']
        seq_x[:, 0, 4] = X_df['incidencia_lag3']
        
        # Paso 2: Hace 2 meses (Lag 2)
        seq_x[:, 1, 0] = X_df['tmax_lag2']
        seq_x[:, 1, 1] = X_df['tmin_lag2']
        seq_x[:, 1, 2] = X_df['precipitacion_lag2']
        seq_x[:, 1, 3] = X_df['humedad_lag2']
        seq_x[:, 1, 4] = X_df['incidencia_lag2']
        
        # Paso 3: Hace 1 mes (Lag 1)
        seq_x[:, 2, 0] = X_df['tmax_lag1']
        seq_x[:, 2, 1] = X_df['tmin_lag1']
        seq_x[:, 2, 2] = X_df['precipitacion_lag1']
        seq_x[:, 2, 3] = X_df['humedad_lag1']
        seq_x[:, 2, 4] = X_df['incidencia_lag1']
        
        # 2. Variables estáticas o del mes objetivo: agua_basica y clima actual
        static_x = X_df[['agua_basica', 'tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']].values
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(static_x, dtype=torch.float32)

    def entrenar_modelo_completo(self, seq_train, static_train, y_train, epochs=25, lr=0.01):
        """
        Entrena el modelo LSTM final de PyTorch.
        """
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        model = DengueLSTMModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(seq_train, static_train)
            loss = criterion(preds, y_train_t)
            loss.backward()
            optimizer.step()
            
        return model

    def entrenar_modelo(self):
        """
        Carga el dataset mensual, realiza la partición cronológica (2014-2020 / 2021-2022)
        y entrena el regresor LSTM en PyTorch con validación cruzada K-Fold y test set.
        """
        print("[Agente 4] Cargando dataset maestro mensual consolidado...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Error: No se encontró el dataset maestro '{self.dataset_path}'.")
            
        df = pd.read_csv(self.dataset_path)
        
        # 1. Definir exclusiones y variables predictoras
        COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        
        # 2. Partición Cronológica
        print("   [DL/LSTM] Particionando datos: Entrenamiento (2014-2020) | Prueba (2021-2022)")
        df_train_raw = df[df['ano'] <= 2020].copy()
        df_test_raw = df[df['ano'] >= 2021].copy()
        
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
        
        # 4. Formatear datos para la LSTM en PyTorch (Secuencias y Estáticos)
        seq_train, static_train = self.preparar_secuencias(X_train_esc)
        seq_test, static_test = self.preparar_secuencias(X_test_esc)
        
        # 5. Validación Cruzada K-Fold (K=5) sobre el bloque de entrenamiento (2014-2020)
        print("   [DL/LSTM] Iniciando validación cruzada K-Fold (K=5) en PyTorch...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.semilla)
        
        cv_mae_list, cv_rmse_list, cv_r2_list = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_esc)):
            # Splits del Fold
            seq_tr, seq_val = seq_train[train_idx], seq_train[val_idx]
            static_tr, static_val = static_train[train_idx], static_train[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Entrenar modelo temporal en el Fold
            m_fold = self.entrenar_modelo_completo(seq_tr, static_tr, y_tr, epochs=20, lr=0.01)
            
            # Evaluar en Validación
            m_fold.eval()
            with torch.no_grad():
                preds_val = m_fold(seq_val, static_val).numpy().flatten()
                preds_val = np.clip(preds_val, 0.0, None)
                
            cv_mae_list.append(mean_absolute_error(y_val, preds_val))
            cv_rmse_list.append(np.sqrt(mean_squared_error(y_val, preds_val)))
            cv_r2_list.append(r2_score(y_val, preds_val))
            
        cv_mae = np.mean(cv_mae_list)
        cv_rmse = np.mean(cv_rmse_list)
        cv_r2 = np.mean(cv_r2_list)
        print(f"   [DL/LSTM] Resultados CV (Train): MAE: {cv_mae:.4f} | RMSE: {cv_rmse:.4f} | R²: {cv_r2*100:.2f}%")
        
        # 6. Entrenamiento final sobre el bloque de Entrenamiento completo (2014-2020)
        print("   [DL/LSTM] Entrenando modelo final LSTM en todo el Train Set...")
        modelo_lstm = self.entrenar_modelo_completo(seq_train, static_train, y_train, epochs=25, lr=0.01)
        
        # 7. Proyección y Evaluación sobre el Conjunto de Prueba Independiente (2021-2022)
        print("   [DL/LSTM] Evaluando LSTM sobre Test Set (2021-2022)...")
        modelo_lstm.eval()
        with torch.no_grad():
            y_pred = modelo_lstm(seq_test, static_test).numpy().flatten()
            y_pred = np.clip(y_pred, 0.0, None)
            
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        print(f"   [DL/LSTM] Resultados Test (21-22): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2*100:.2f}%")
        
        print("SUCCESS: [Agente 4] Inferencia secuencial profunda LSTM completada.")
        print("="*70)
        
        return {
            "modelo": modelo_lstm,
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
            "y_pred": y_pred,
            "preparar_secuencias_fn": self.preparar_secuencias
        }

if __name__ == "__main__":
    agente = AgentePrediccionDL()
    resultados = agente.entrenar_modelo()
