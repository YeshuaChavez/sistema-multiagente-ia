# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 2: Preprocesamiento
--------------------------------------------------
Responsabilidad: Limpieza, transformación, unión de fuentes multidominio,
normalización de tasas de incidencia y estructuración de rezagos simétricos (lags 1, 2 y 3).
Produce el dataset maestro mensual final del proyecto.
"""

import os
import sys
import pandas as pd
import numpy as np

class AgentePreprocesamiento:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir
            
        self.db_dir = os.path.join(self.base_dir, "Base de Datos")
        self.output_path = os.path.join(self.db_dir, "dataset_maestro_mensual_latam.csv")

    def procesar_casos(self, df_casos):
        """
        Agrupa los casos de dengue por departamento, año y mes.
        """
        print("[Agente 2] Procesando y agregando casos epidemiológicos a nivel mensual...")
        df_casos = df_casos.copy()
        
        # Estandarizar nombres de columnas
        df_casos['iso_a0'] = df_casos['ISO_A0']
        
        # Parsear fecha de inicio y extraer año y mes
        df_casos['date'] = pd.to_datetime(df_casos['calendar_start_date'])
        df_casos['ano'] = df_casos['date'].dt.year
        df_casos['mes'] = df_casos['date'].dt.month
        
        # Agrupar a resolución mensual
        df_monthly_cases = df_casos.groupby(
            ['iso_a0', 'adm_1_name', 'ano', 'mes'], as_index=False
        )['dengue_total'].sum()
        
        df_monthly_cases.rename(columns={'dengue_total': 'casos_dengue'}, inplace=True)
        print(f"   [Casos] Agrupación mensual completada. Shape: {df_monthly_cases.shape}")
        return df_monthly_cases

    def fusionar_datos(self, df_casos_m, df_clima, df_agua, df_poblacion):
        """
        Une todas las fuentes de datos en base a las claves geográficas y temporales.
        """
        print("[Agente 2] Iniciando fusión multidominio de datos...")
        
        # Estandarizar nombres y tipos
        df_casos_m = df_casos_m.copy()
        df_clima = df_clima.copy()
        df_agua = df_agua.copy()
        df_poblacion = df_poblacion.copy()
        
        # Asegurar tipos
        for df in [df_casos_m, df_clima, df_agua]:
            df['ano'] = df['ano'].astype(int)
            df['mes'] = df['mes'].astype(int)
            df['iso_a0'] = df['iso_a0'].astype(str).str.strip().str.upper()
            df['adm_1_name'] = df['adm_1_name'].astype(str).str.strip()
            
        df_poblacion['ano'] = df_poblacion['ano'].astype(int)
        df_poblacion['iso_a0'] = df_poblacion['iso_a0'].astype(str).str.strip().str.upper()
        df_poblacion['adm_1_name'] = df_poblacion['adm_1_name'].astype(str).str.strip()

        # 1. Cruzar casos con clima
        print("   [Fusión] Cruzando Casos + Clima...")
        df_merged = pd.merge(
            df_clima, df_casos_m, 
            on=['iso_a0', 'adm_1_name', 'ano', 'mes'], 
            how='left'
        )
        
        # Casos vacíos se asumen como cero
        df_merged['casos_dengue'] = df_merged['casos_dengue'].fillna(0).astype(int)

        # 2. Cruzar con agua potable
        print("   [Fusión] Cruzando con Acceso a Agua...")
        df_merged = pd.merge(
            df_merged, df_agua[['iso_a0', 'adm_1_name', 'ano', 'mes', 'agua_basica']], 
            on=['iso_a0', 'adm_1_name', 'ano', 'mes'], 
            how='left'
        )
        # Rellenar agua básica si falta usando interpolación o la mediana del departamento
        df_merged['agua_basica'] = df_merged.groupby(['iso_a0', 'adm_1_name'])['agua_basica'].transform(
            lambda x: x.ffill().bfill()
        )
        # Si aún falta agua básica, rellenar con la mediana nacional
        df_merged['agua_basica'] = df_merged.groupby(['iso_a0'])['agua_basica'].transform(
            lambda x: x.fillna(x.median())
        )

        # 3. Cruzar con poblaciones subnacionales de censos
        print("   [Fusión] Cruzando con Población de Censos Gubernamentales...")
        df_merged = pd.merge(
            df_merged, df_poblacion[['iso_a0', 'adm_1_name', 'ano', 'poblacion']], 
            on=['iso_a0', 'adm_1_name', 'ano'], 
            how='inner'
        )
        
        print(f"   [Fusión] Registros integrados: {df_merged.shape[0]} observaciones.")
        return df_merged

    def calcular_incidencia_y_rezagos(self, df_merged):
        """
        Calcula la tasa de incidencia de dengue y estructura los rezagos simétricos (lags 1, 2 y 3).
        """
        print("[Agente 2] Calculando tasas de incidencia y variables rezagadas simétricas (Lags 1-3)...")
        df_merged = df_merged.copy()
        
        # 1. Calcular incidencia normalizada por cada 100k hab.
        df_merged['incidencia_dengue'] = (df_merged['casos_dengue'] / df_merged['poblacion']) * 100000
        df_merged['incidencia_dengue'] = df_merged['incidencia_dengue'].round(4)
        
        # 2. Ordenar cronológicamente para el cálculo de lags
        df_merged = df_merged.sort_values(['pais', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        group = df_merged.groupby(['pais', 'adm_1_name'])
        
        # 3. Lags climáticos simétricos (lags 1, 2, 3)
        cols_clima = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
        for var in cols_clima:
            base_name = var.split('_')[0] if 'promedio' in var else var
            for lag in [1, 2, 3]:
                df_merged[f"{base_name}_lag{lag}"] = group[var].shift(lag)
                
        # 4. Lags autorregresivos (lags 1, 2, 3)
        for lag in [1, 2, 3]:
            df_merged[f"incidencia_lag{lag}"] = group['incidencia_dengue'].shift(lag)
            
        # 5. Limpieza de filas con lags nulos (solo los primeros 3 meses de 2014)
        print("   [Lags] Limpiando nulos iniciales de rezago temporal...")
        cols_lags = [c for c in df_merged.columns if 'lag' in c]
        df_merged.dropna(subset=cols_lags, inplace=True)
        df_merged = df_merged.reset_index(drop=True)
        
        # Estandarizar el esquema final de columnas (28 columnas en total)
        base_cols = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 'incidencia_dengue', 'agua_basica', 'tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
        climate_lag_cols = []
        for var in ['tmax', 'tmin', 'precipitacion', 'humedad']:
            for lag in [1, 2, 3]:
                climate_lag_cols.append(f"{var}_lag{lag}")
        epi_lag_cols = [f"incidencia_lag{lag}" for lag in [1, 2, 3]]
        pop_cols = ['poblacion']
        
        final_columns = base_cols + climate_lag_cols + epi_lag_cols + pop_cols
        df_final = df_merged[final_columns]
        
        print(f"   [Lags] Procesamiento completado. Columnas finales: {len(df_final.columns)} | Registros: {df_final.shape[0]}")
        return df_final

    def ejecutar_preprocesamiento(self, datos_crudos):
        """
        Ejecuta el pipeline completo de preprocesamiento y guarda el dataset maestro resultante.
        """
        print("="*70)
        print("  EJECUTANDO PREPROCESAMIENTO — AGENTE 2: PREPROCESAMIENTO")
        print("="*70)
        
        # 1. Procesar casos
        df_casos_m = self.procesar_casos(datos_crudos['casos'])
        
        # 2. Fusionar
        df_merged = self.fusionar_datos(
            df_casos_m, 
            datos_crudos['clima'], 
            datos_crudos['agua'], 
            datos_crudos['poblacion']
        )
        
        # 3. Incidencia y rezagos
        df_final = self.calcular_incidencia_y_rezagos(df_merged)
        
        # 4. Guardar archivo maestro mensual consolidado
        df_final.to_csv(self.output_path, index=False)
        print(f"SUCCESS: [Agente 2] Dataset maestro mensual guardado en: {self.output_path}")
        print("="*70)
        
        return df_final

if __name__ == "__main__":
    from agente_1_recoleccion import AgenteRecoleccion
    recolector = AgenteRecoleccion()
    datos_crudos = recolector.ejecutar_ingesta()
    
    preprocesador = AgentePreprocesamiento()
    df_maestro = preprocesador.ejecutar_preprocesamiento(datos_crudos)
