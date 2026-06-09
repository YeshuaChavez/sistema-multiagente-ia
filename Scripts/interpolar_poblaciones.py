# -*- coding: utf-8 -*-
"""
Script de Interpolación de Poblaciones Subnacionales (2014-2022)
Proyecto Final - IA
----------------------------------------------------------------
Este script lee los archivos de censos crudos por país y genera
las poblaciones anuales interpoladas linealmente de 2014 a 2022.
Garantiza trazabilidad y reproducibilidad para el proyecto.
"""

import os
import pandas as pd
import numpy as np

# Rutas
base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
raw_pob_dir = os.path.join(base_dir, "Base de Datos", "datos_crudos", "poblacion")
proc_pob_dir = os.path.join(base_dir, "Base de Datos", "datos_procesados", "poblacion")

countries = ['colombia', 'argentina', 'brazil', 'ecuador', 'mexico', 'panama', 'bolivia', 'nicaragua', 'peru']
target_years = list(range(2014, 2025)) # 2014 a 2024 inclusive

def interpolar_linear(t, t_coords, p_coords):
    """
    Realiza interpolación lineal (o extrapolación si está fuera de rango)
    para el año t dados los puntos de censos (t_coords) y sus poblaciones (p_coords).
    """
    t_coords = np.array(t_coords)
    p_coords = np.array(p_coords)
    
    # Ordenar por año
    sort_idx = np.argsort(t_coords)
    t_coords = t_coords[sort_idx]
    p_coords = p_coords[sort_idx]
    
    # Si t coincide exactamente con un año censal
    if t in t_coords:
        return p_coords[np.where(t_coords == t)[0][0]]
        
    # Si t está entre dos censos (Interpolación)
    for i in range(len(t_coords) - 1):
        t1, t2 = t_coords[i], t_coords[i+1]
        p1, p2 = p_coords[i], p_coords[i+1]
        if t1 < t < t2:
            val = p1 + ((p2 - p1) / (t2 - t1)) * (t - t1)
            return int(round(val))
            
    # Si t está antes del primer censo (Extrapolación hacia atrás)
    if t < t_coords[0]:
        # Usar los dos primeros censos para la pendiente
        t1, t2 = t_coords[0], t_coords[1]
        p1, p2 = p_coords[0], p_coords[1]
        val = p1 + ((p2 - p1) / (t2 - t1)) * (t - t1)
        return int(round(val))
        
    # Si t está después del último censo (Extrapolación hacia adelante)
    if t > t_coords[-1]:
        # Usar los dos últimos censos para la pendiente
        t1, t2 = t_coords[-2], t_coords[-1]
        p1, p2 = p_coords[-2], p_coords[-1]
        val = p1 + ((p2 - p1) / (t2 - t1)) * (t - t1)
        return int(round(val))
        
    return 0

def procesar_pais(country):
    print(f"Interpolando población para {country.upper()}...")
    raw_path = os.path.join(raw_pob_dir, f"censos_crudos_{country}.csv")
    out_path = os.path.join(proc_pob_dir, f"poblacion_{country}.csv")
    
    if not os.path.exists(raw_path):
        print(f"   [Error] No se encontró el archivo de censo crudo: {raw_path}")
        return
        
    df_raw = pd.read_csv(raw_path)
    
    # Caso especial: Perú
    if country == 'peru':
        # Perú tiene serie anual de estimaciones de 2014 a 2022
        # Haremos una regresión lineal por departamento para estimar 2010-2013,
        # pero conservando los datos anuales oficiales originales de 2014-2022.
        records = []
        for dept in df_raw['adm_1_name'].unique():
            df_dept = df_raw[df_raw['adm_1_name'] == dept].sort_values('anio')
            X = df_dept['anio'].values # 2014-2022
            y = df_dept['poblacion'].values
            
            # Ajustar tendencia lineal
            slope, intercept = np.polyfit(X, y, 1)
            
            for yr in target_years:
                if yr in X:
                    # Usar el valor original de la serie
                    pob_val = df_dept[df_dept['anio'] == yr]['poblacion'].values[0]
                else:
                    # Extrapolar linealmente hacia atrás
                    pob_val = slope * yr + intercept
                    pob_val = int(round(pob_val))
                    
                records.append({
                    'pais': 'Peru',
                    'adm_1_name': dept,
                    'ano': yr,
                    'poblacion': max(1, pob_val)
                })
        df_out = pd.DataFrame(records)
    else:
        # Países con censos por años
        records = []
        pais_name = df_raw['pais'].iloc[0]
        for dept in df_raw['adm_1_name'].unique():
            df_dept = df_raw[df_raw['adm_1_name'] == dept]
            t_coords = df_dept['anio'].tolist()
            p_coords = df_dept['poblacion'].tolist()
            
            for yr in target_years:
                pob_val = interpolar_linear(yr, t_coords, p_coords)
                records.append({
                    'pais': pais_name,
                    'adm_1_name': dept,
                    'ano': yr,
                    'poblacion': max(1, pob_val)
                })
        df_out = pd.DataFrame(records)
        
    df_out = df_out.sort_values(by=['adm_1_name', 'ano']).reset_index(drop=True)
    df_out.to_csv(out_path, index=False)
    print(f"   [Éxito] Generado: {out_path} ({len(df_out)} filas, años {min(df_out['ano'])}-{max(df_out['ano'])})")

def main():
    print("="*60)
    print("  EJECUTANDO INTERPOLACIÓN DE POBLACIONES SUBNACOINALES")
    print("="*60)
    for c in countries:
        procesar_pais(c)
    print("="*60)
    print("PROCESO COMPLETADO CON ÉXITO")

if __name__ == "__main__":
    main()
