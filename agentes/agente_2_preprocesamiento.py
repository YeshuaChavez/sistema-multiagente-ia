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
        self.output_path = os.path.join(self.db_dir, "datos_procesados", "dataset_maestro_mensual_latam.csv")

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
        Calcula la tasa de incidencia de dengue y limpia el dataset mensual.
        En la Opción B, no precalculamos rezagos en disco.
        """
        print("[Agente 2] Calculando tasas de incidencia de dengue...")
        df_merged = df_merged.copy()
        
        # 1. Calcular incidencia normalizada por cada 100k hab.
        df_merged['incidencia_dengue'] = (df_merged['casos_dengue'] / df_merged['poblacion']) * 100000
        df_merged['incidencia_dengue'] = df_merged['incidencia_dengue'].round(4)
        
        # 2. Cargar las áreas departamentales y calcular la densidad poblacional
        areas_path = os.path.join(self.db_dir, "datos_crudos", "departamentos_areas.csv")
        if os.path.exists(areas_path):
            print("   [Preprocesamiento] Integrando áreas departamentales y calculando densidad poblacional...")
            df_areas = pd.read_csv(areas_path)
            # Estandarizar
            df_areas['iso_a0'] = df_areas['iso_a0'].astype(str).str.strip().str.upper()
            df_areas['adm_1_name'] = df_areas['adm_1_name'].astype(str).str.strip()
            
            # Cruzar
            df_merged = pd.merge(df_merged, df_areas[['iso_a0', 'adm_1_name', 'area_km2']], on=['iso_a0', 'adm_1_name'], how='left')
            # Calcular densidad (Población / Área)
            df_merged['densidad_poblacion'] = df_merged['poblacion'] / df_merged['area_km2']
            df_merged['densidad_poblacion'] = df_merged['densidad_poblacion'].round(4)
            # Rellenar con la mediana si hay algún nulo (no debería)
            df_merged['densidad_poblacion'] = df_merged['densidad_poblacion'].fillna(df_merged['densidad_poblacion'].median())
        else:
            print("   [Advertencia] No se encontró el archivo de áreas departamentales. Densidad = 0.0")
            df_merged['densidad_poblacion'] = 0.0
            
        # 3. Ordenar cronológicamente
        df_final = df_merged.sort_values(['pais', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        
        # Estandarizar el esquema final de columnas (14 columnas en total)
        final_columns = [
            'iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue', 
            'incidencia_dengue', 'agua_basica', 'tmax_promedio', 'tmin_promedio', 
            'precipitacion', 'humedad_promedio', 'poblacion', 'densidad_poblacion'
        ]
        df_final = df_final[final_columns]
        
        print(f"   [Preprocesamiento] Completado. Columnas finales: {len(df_final.columns)} | Registros: {df_final.shape[0]}")
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
        
        # Filtrar estrictamente hasta el año 2022
        print("   [Preprocesamiento] Truncando dataset maestro final al rango 2014-2022...")
        df_final = df_final[df_final['ano'] <= 2022].reset_index(drop=True)
        
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
