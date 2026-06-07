import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("  Compilación del Dataset Maestro Unificado (LATAM)")
    print("=" * 60)
    
    # Rutas de archivos de entrada
    dengue_path = "Base de Datos/dengue_mensual_casos.csv"
    population_path = "Scripts/poblacion_latam_2000_2024.csv"
    water_path = "Scripts/agua_latam_detallado.csv"
    climate_path = "Scripts/nasa_power_latam_mensual.csv"
    
    output_path = "Base de Datos/dataset_maestro_latam.csv"
    
    print(f"1. Cargando casos de dengue mensuales: {dengue_path}...")
    df_dengue = pd.read_csv(dengue_path)
    
    print(f"2. Cargando datos de población anuales: {population_path}...")
    df_pop = pd.read_csv(population_path)
    
    print(f"3. Cargando datos detallados de agua (JMP): {water_path}...")
    df_water = pd.read_csv(water_path)
    
    print(f"4. Cargando datos climáticos mensuales: {climate_path}...")
    df_climate = pd.read_csv(climate_path)
    
    # 20 Países Latinoamericanos seleccionados
    PAISES_ISO = [
        "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
        "HTI", "HND", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY", "VEN"
    ]
    
    # Filtrar clima solo para los 20 países de interés
    df_climate = df_climate[df_climate['iso_a0'].isin(PAISES_ISO)].copy()
    
    print("\nCalculando lags mensuales para las variables climáticas...")
    # Ordenar variables de clima para calcular los rezagos correctamente
    df_climate = df_climate.sort_values(['iso_a0', 'ano', 'mes']).reset_index(drop=True)
    
    # Definir variables climáticas a rezagar
    climate_vars = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
    lags = [1, 2, 3, 6]
    
    # Generar las columnas de rezagos (lags)
    for var in climate_vars:
        for lag in lags:
            col_name = f"{var.split('_')[0] if 'promedio' in var else var}_lag{lag}"
            df_climate[col_name] = df_climate.groupby('iso_a0')[var].shift(lag)
            
    print("Rezagos climáticos calculados con éxito.")
    
    # Unir dengue con población
    print("\nRealizando el merge de dengue con población...")
    df_master = pd.merge(df_dengue, df_pop, on=['iso_a0', 'ano'], how='inner')
    
    # Unir con agua y saneamiento
    print("Realizando el merge con indicadores de agua detallados...")
    df_master = pd.merge(df_master, df_water, on=['iso_a0', 'ano'], how='inner')
    
    # Unir con clima y sus rezagos
    print("Realizando el merge con variables climáticas y sus rezagos...")
    df_master = pd.merge(df_master, df_climate, on=['iso_a0', 'ano', 'mes'], how='inner')
    
    # Calcular Tasa de Incidencia Mensual de Dengue (casos por cada 100,000 habitantes)
    print("\nCalculando la tasa de incidencia mensual de dengue...")
    df_master['incidencia_dengue'] = (df_master['casos'] / df_master['poblacion']) * 100000
    df_master['incidencia_dengue'] = df_master['incidencia_dengue'].round(4)
    
    # Renombrar columna de casos para mayor claridad
    df_master.rename(columns={'casos': 'casos_dengue'}, inplace=True)
    
    # Quitar duplicados o columnas de nombre repetidas
    # Si existía la columna 'pais' en clima y dengue, remover la redundante (nos quedamos con la del dengue)
    if 'pais_y' in df_master.columns:
        df_master.drop(columns=['pais_y'], inplace=True)
    if 'pais_x' in df_master.columns:
        df_master.rename(columns={'pais_x': 'pais'}, inplace=True)
        
    # Ordenar columnas lógicamente
    cols_base = ['iso_a0', 'pais', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']
    cols_water = ['agua_basica', 'agua_limitada', 'agua_no_mejorada', 'agua_superficial', 
                  'agua_entubada', 'agua_no_entubada', 'agua_disponible', 'agua_segura']
    cols_climate = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
    cols_lags = [col for col in df_master.columns if 'lag' in col]
    
    final_cols = cols_base + cols_water + cols_climate + cols_lags
    df_master = df_master[final_cols].copy()
    
    # Auditar la presencia de valores nulos antes de filtrar
    print(f"\nAuditoría antes de limpiar lags vacíos:")
    print(f"Total de registros iniciales: {len(df_master)}")
    print(f"Registros con lags nulos: {df_master[cols_lags].isnull().any(axis=1).sum()}")
    
    # Filtrar registros donde los lags climáticos son nulos (primeros meses de la serie climática en 2000)
    # Esto garantiza 0 nulos en las variables predictoras clave
    df_master = df_master.dropna(subset=cols_lags).reset_index(drop=True)
    
    # Guardar a CSV
    df_master.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"Dataset Maestro generado -> {output_path}")
    print(f"Filas totales            -> {len(df_master)}")
    print(f"Columnas totales         -> {len(df_master.columns)}")
    print(f"Países incluidos         -> {df_master['iso_a0'].nunique()}")
    print(f"Valores nulos en dataset -> {df_master.isnull().sum().sum()}")
    print("=" * 60)
    
    # Imprimir resumen de nulos por columna para verificar que es 0
    print("\nConteo de valores nulos por columna en el archivo final:")
    null_counts = df_master.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  - {col}: {count} nulos")
    if null_counts.sum() == 0:
        print("  ¡Excelente! Todas las columnas tienen 0 valores nulos.")
        
    print("\nVisualización de las primeras 3 filas del dataset:")
    print(df_master.head(3).to_string())

if __name__ == "__main__":
    main()
