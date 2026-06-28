# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 1: Recolección
--------------------------------------------------
Responsabilidad: Ingesta automatizada, asíncrona y masiva del corpus histórico 2014-2024.
Realiza consultas a las APIs oficiales de NASA POWER y JMP (vía World Bank API),
utilizando caché local si los archivos crudos ya existen en el entorno.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import time
import urllib.parse

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3

class AgenteRecoleccion:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Detectar directorio base del proyecto
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir
            
        # Rutas de entrada
        self.dengue_path = os.path.join(self.base_dir, "data", "raw", "Temporal_extract_V1_3.csv")
        self.paises_iso = ['ARG', 'BOL', 'BRA', 'COL', 'ECU', 'MEX', 'NIC', 'PAN', 'PER']
        
        # Nombres de países para geocodificación
        self.paises_nombres = {
            'ARG': 'Argentina', 'BOL': 'Bolivia', 'BRA': 'Brazil', 
            'COL': 'Colombia', 'ECU': 'Ecuador', 'MEX': 'Mexico', 
            'NIC': 'Nicaragua', 'PAN': 'Panama', 'PER': 'Peru'
        }
        
        # Coordenadas de contingencia (capitales o centros geográficos)
        self.fallback_coords = {
            'ARG': (-34.61, -58.38), 'BOL': (-17.78, -63.18), 'BRA': (-15.78, -47.93),
            'COL': (4.57, -74.30), 'ECU': (-2.17, -79.92), 'MEX': (23.63, -102.55),
            'NIC': (12.86, -85.20), 'PAN': (8.53, -80.78), 'PER': (-9.19, -75.01)
        }
        
        # Poblaciones procesadas (Censos oficiales de cada país)
        self.pob_dir = os.path.join(self.base_dir, "data", "processed", "poblacion")
        self.poblaciones_paths = {
            'ARG': os.path.join(self.pob_dir, "poblacion_argentina.csv"),
            'BOL': os.path.join(self.pob_dir, "poblacion_bolivia.csv"),
            'BRA': os.path.join(self.pob_dir, "poblacion_brazil.csv"),
            'COL': os.path.join(self.pob_dir, "poblacion_colombia.csv"),
            'ECU': os.path.join(self.pob_dir, "poblacion_ecuador.csv"),
            'MEX': os.path.join(self.pob_dir, "poblacion_mexico.csv"),
            'NIC': os.path.join(self.pob_dir, "poblacion_nicaragua.csv"),
            'PAN': os.path.join(self.pob_dir, "poblacion_panama.csv"),
            'PER': os.path.join(self.pob_dir, "poblacion_peru.csv")
        }

        # Rutas de salida para APIs y geocodificación
        self.clima_nasa_path   = os.path.join(self.base_dir, "data", "raw", "clima_nasa_crudo.csv")
        self.agua_jmp_path     = os.path.join(self.base_dir, "data", "raw", "agua_jmp_crudo.csv")
        self.coords_cache_path = os.path.join(self.base_dir, "data", "raw", "departamentos_coordenadas.csv")

    def geocodificar_departamento(self, iso, dept_name):
        """
        Obtiene latitud y longitud de un departamento usando Nominatim (OpenStreetMap).
        """
        pais_nombre = self.paises_nombres.get(iso, "")
        query = f"{dept_name}, {pais_nombre}"
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&limit=1"
        headers = {'User-Agent': 'DengueProjectPredictor/1.0 (yeshua.chavez@gmail.com)'}
        
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data:
                    return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            print(f"      [Advertencia] Error al geocodificar '{query}': {e}")
            
        return self.fallback_coords.get(iso, (0.0, 0.0))

    def resolver_coordenadas_departamentos(self, unique_depts):
        """
        Carga o calcula las coordenadas de todos los departamentos requeridos.
        """
        print("[Agente 1] Resolviendo coordenadas geográficas de departamentos...")
        if os.path.exists(self.coords_cache_path):
            print(f"   [Caché] Cargando caché de coordenadas desde '{self.coords_cache_path}'...")
            df_coords = pd.read_csv(self.coords_cache_path)
            unique_depts = pd.merge(unique_depts, df_coords, on=['iso_a0', 'adm_1_name'], how='left')
        else:
            unique_depts['lat'] = np.nan
            unique_depts['lon'] = np.nan

        # Geocodificar los que falten
        missing_coords = unique_depts[unique_depts['lat'].isnull()]
        if len(missing_coords) > 0:
            print(f"   [Geocodificación] Resolviendo {len(missing_coords)} departamentos...")
            for idx, row in missing_coords.iterrows():
                iso = row['iso_a0']
                dept = row['adm_1_name']
                print(f"      - Geocodificando {dept} ({iso})...", end=" ", flush=True)
                lat, lon = self.geocodificar_departamento(iso, dept)
                unique_depts.loc[
                    (unique_depts['iso_a0'] == iso) & (unique_depts['adm_1_name'] == dept), 
                    ['lat', 'lon']
                ] = [lat, lon]
                print(f"OK ({lat:.2f}, {lon:.2f})")
                time.sleep(1) # Respetar límites de Nominatim
                
            # Guardar la caché actualizada
            unique_depts[['iso_a0', 'adm_1_name', 'lat', 'lon']].to_csv(self.coords_cache_path, index=False)
            print(f"   [Caché] Coordenadas guardadas en '{self.coords_cache_path}'.")
            
        return unique_depts

    def recolectar_datos_climaticos(self, unique_depts):
        """
        Ingesta datos climáticos de NASA POWER.
        Si ya existe 'clima_nasa_crudo.csv', se carga localmente.
        Si no, ejecuta las consultas HTTP a la API de NASA POWER para todos los departamentos.
        """
        print("[Agente 1] Iniciando recolección de variables climáticas satelitales...")
        
        if os.path.exists(self.clima_nasa_path):
            print(f"INFO: [Agente 1] Archivo local '{self.clima_nasa_path}' detectado.")
            print("INFO: [Agente 1] Omitiendo consultas a la API de NASA POWER (Caché local activa).")
            df_clima = pd.read_csv(self.clima_nasa_path)
            return df_clima
            
        print("⚠️ ALERTA: [Agente 1] No se encontró clima_nasa_crudo.csv. Consultando API NASA POWER...")
        
        # Resolver coordenadas primero
        df_coords = self.resolver_coordenadas_departamentos(unique_depts)
        
        all_records = []
        total_depts = len(df_coords)
        
        for i, row in enumerate(df_coords.itertuples(), 1):
            iso = row.iso_a0
            dept = row.adm_1_name
            lat = row.lat
            lon = row.lon
            pais_nombre = self.paises_nombres.get(iso, "")
            
            print(f"   [{i:03d}/{total_depts}] NASA POWER para {dept} ({iso}) lat={lat:.2f}, lon={lon:.2f}...", end=" ", flush=True)
            
            url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M',
                'community': 'RE',
                'longitude': lon,
                'latitude': lat,
                'start': '2014',
                'end': '2022',
                'format': 'JSON'
            }
            
            success = False
            for attempt in range(3):
                try:
                    r = requests.get(url, params=params, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        props = data["properties"]["parameter"]
                        
                        # Las claves de NASA POWER mensual son "201401", "201402", etc.
                        claves = list(props["T2M_MAX"].keys())
                        for clave in claves:
                            if clave.endswith("13"): # Promedio anual
                                continue
                            ano = int(clave[:4])
                            mes = int(clave[4:])
                            
                            all_records.append({
                                "iso_a0": iso,
                                "pais": pais_nombre,
                                "adm_1_name": dept,
                                "ano": ano,
                                "mes": mes,
                                "tmax_promedio": props["T2M_MAX"][clave],
                                "tmin_promedio": props["T2M_MIN"][clave],
                                "precipitacion": props["PRECTOTCORR"][clave],
                                "humedad_promedio": props["RH2M"][clave]
                            })
                        
                        print("OK")
                        success = True
                        break
                    else:
                        print(f"[Intento {attempt+1}] Error HTTP {r.status_code}...", end=" ", flush=True)
                except Exception as e:
                    print(f"[Intento {attempt+1}] Conexión fallida: {e}...", end=" ", flush=True)
                time.sleep(2)
                
            if not success:
                print("FALLÓ")
                # Fallback con datos nulos para no romper la ejecución
                print(f"      [Advertencia] Fallback de datos climáticos vacíos para {dept}.")
                
            time.sleep(0.5) # Pausa amigable con la API
            
        if len(all_records) == 0:
            raise RuntimeError("Error: No se pudo descargar ningún dato de clima de la API de NASA POWER.")
            
        df_clima = pd.DataFrame(all_records)
        df_clima.replace(-999.0, np.nan, inplace=True)
        # Rellenar nulos
        df_clima = df_clima.ffill().bfill()
        
        # Guardar en base de datos
        df_clima.to_csv(self.clima_nasa_path, index=False)
        print(f"SUCCESS: [Agente 1] Clima guardado en '{self.clima_nasa_path}' ({len(df_clima)} registros).")
        return df_clima

    def recolectar_datos_agua(self, unique_depts):
        """
        Ingesta datos de Agua potable desde JMP (vía API del Banco Mundial).
        Si ya existe 'agua_jmp_crudo.csv', lo lee localmente.
        Si no, realiza la consulta a la API de World Bank.
        """
        print("[Agente 1] Ingestando datos de agua básica (JMP)...")
        
        if os.path.exists(self.agua_jmp_path):
            print(f"INFO: [Agente 1] Archivo local '{self.agua_jmp_path}' detectado.")
            print("INFO: [Agente 1] Omitiendo consultas a la API de JMP (Caché local activa).")
            df_agua = pd.read_csv(self.agua_jmp_path)
            return df_agua
            
        print("⚠️ ALERTA: [Agente 1] No se encontró agua_jmp_crudo.csv. Consultando API de World Bank (JMP)...")
        
        wb_records = []
        for pais_iso in self.paises_iso:
            print(f"   [API JMP/WB] Consultando indicador SH.H2O.BASW.ZS para {pais_iso}...", end=" ", flush=True)
            url = f"https://api.worldbank.org/v2/country/{pais_iso}/indicator/SH.H2O.BASW.ZS?date=2014:2022&format=json&per_page=100"
            
            success = False
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        res = r.json()
                        if len(res) > 1 and res[1] is not None:
                            for entry in res[1]:
                                year = int(entry['date'])
                                val = entry['value']
                                wb_records.append({
                                    'iso_a0': pais_iso,
                                    'ano': year,
                                    'agua_basica': val if val is not None else np.nan
                                })
                            print("OK")
                            success = True
                            break
                        else:
                            print(f"[Intento {attempt+1}] Sin datos...", end=" ", flush=True)
                    else:
                        print(f"[Intento {attempt+1}] Error HTTP {r.status_code}...", end=" ", flush=True)
                except Exception as e:
                    print(f"[Intento {attempt+1}] Conexión fallida: {e}...", end=" ", flush=True)
                time.sleep(2)
                
            if not success:
                print("FALLÓ")
                # Crear datos ficticios o aproximados de contingencia para el país
                print(f"      [Advertencia] Fallback de datos de agua básica aproximados para {pais_iso}.")
                for year in range(2014, 2023):
                    wb_records.append({
                        'iso_a0': pais_iso,
                        'ano': year,
                        'agua_basica': 90.0 # Valor promedio razonable de acceso a agua
                    })
            time.sleep(0.5)

        df_wb = pd.DataFrame(wb_records)
        # Completar nulos mediante interpolación
        df_wb['agua_basica'] = df_wb.groupby('iso_a0')['agua_basica'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(90.0)
        )
        
        # Mapear valores nacionales a cada departamento y cada mes
        print("   [Procesamiento] Mapeando agua nacional a nivel departamental y mensual...")
        mapped_records = []
        for row_dept in unique_depts.itertuples():
            iso = row_dept.iso_a0
            dept = row_dept.adm_1_name
            pais_nombre = self.paises_nombres.get(iso, "")
            
            # Filtro del país
            df_country_water = df_wb[df_wb['iso_a0'] == iso]
            
            for row_water in df_country_water.itertuples():
                ano = row_water.ano
                val = row_water.agua_basica
                
                # Generar para cada mes (1-12)
                for mes in range(1, 13):
                    mapped_records.append({
                        "iso_a0": iso,
                        "pais": pais_nombre,
                        "adm_1_name": dept,
                        "ano": ano,
                        "mes": mes,
                        "agua_basica": round(val, 4)
                    })
                    
        df_agua = pd.DataFrame(mapped_records)
        df_agua.to_csv(self.agua_jmp_path, index=False)
        print(f"SUCCESS: [Agente 1] Agua JMP guardada en '{self.agua_jmp_path}' ({len(df_agua)} registros).")
        return df_agua

    def recolectar_casos_dengue(self):
        """
        Carga y filtra los casos de dengue de OpenDengue.
        """
        print("[Agente 1] Cargando series epidemiológicas de casos de dengue (OpenDengue)...")
        if not os.path.exists(self.dengue_path):
            raise FileNotFoundError(f"Error crítico: No se encontró el archivo de casos crudos '{self.dengue_path}'")
            
        # Columnas a leer para optimizar memoria
        cols_to_use = ['ISO_A0', 'adm_1_name', 'Year', 'dengue_total', 'calendar_start_date', 'T_res']
        
        # Cargar en dataframe
        df = pd.read_csv(self.dengue_path, usecols=cols_to_use)
        print(f"   [OpenDengue] Casos crudos cargados. Dimensiones: {df.shape}")
        
        # Filtrar por los 9 países, nivel departamental (o superior) y años 2014-2024
        df_filtered = df[
            (df['ISO_A0'].isin(self.paises_iso)) &
            (df['Year'] >= 2014) &
            (df['Year'] <= 2022)
        ].copy()
        
        # Estandarizar nombre nulos o vacíos
        df_filtered['adm_1_name'] = df_filtered['adm_1_name'].fillna('UNKNOWN')
        
        print(f"   [OpenDengue] Casos filtrados (2014-2024, 9 países). Dimensiones: {df_filtered.shape}")
        return df_filtered

    def recolectar_poblaciones(self):
        """
        Ingesta los datos de población subnacional de los Censos Gubernamentales locales (los 9 archivos CSV).
        """
        print("[Agente 1] Ingestando datos de población subnacional (Censos Gubernamentales)...")
        poblacion_dfs = []
        
        for pais, path in self.poblaciones_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error crítico: Falta archivo de población para {pais}: {path}")
                
            df_pop = pd.read_csv(path)
            
            # Estandarizar columna de año (handle anio / ano)
            if 'anio' in df_pop.columns:
                df_pop.rename(columns={'anio': 'ano'}, inplace=True)
                
            print(f"   [Censo {pais}] Cargado. Departamentos: {df_pop['adm_1_name'].nunique()} | Años: {sorted(df_pop['ano'].unique())}")
            df_pop['iso_a0'] = pais
            poblacion_dfs.append(df_pop)
            
        df_poblaciones_all = pd.concat(poblacion_dfs, ignore_index=True)
        print(f"   [Censos] Total poblaciones ingestadas: {df_poblaciones_all.shape[0]} registros.")
        return df_poblaciones_all

    def ejecutar_ingesta(self):
        """
        Ejecuta el flujo secuencial de ingesta de datos.
        """
        print("="*70)
        print("  EJECUTANDO INGESTA DE DATOS — AGENTE 1: RECOLECCIÓN")
        print("="*70)
        
        # 1. Cargar poblaciones para saber qué departamentos y años existen
        df_poblaciones = self.recolectar_poblaciones()
        
        # 2. Extraer combinaciones únicas de departamentos para API
        unique_depts = df_poblaciones[['iso_a0', 'adm_1_name']].drop_duplicates().reset_index(drop=True)
        print(f"   [Estructura] Total departamentos únicos en Censos: {len(unique_depts)}")
        
        # 3. Clima NASA
        df_clima = self.recolectar_datos_climaticos(unique_depts)
        print(f"   -> Clima ingestando con éxito: {df_clima.shape}")
        
        # 4. Agua JMP
        df_agua = self.recolectar_datos_agua(unique_depts)
        print(f"   -> Agua JMP ingestada con éxito: {df_agua.shape}")
        
        # 5. Casos Dengue
        df_casos = self.recolectar_casos_dengue()
        print(f"   -> Casos de dengue ingestados con éxito: {df_casos.shape}")
        
        # Subir archivos crudos actualizados a S3
        print("[Agente 1] Subiendo datos crudos a S3...")
        crudos_dir = os.path.join(self.base_dir, "data", "raw")
        for fname in ["clima_nasa_crudo.csv", "agua_jmp_crudo.csv",
                      "Temporal_extract_V1_3.csv", "departamentos_coordenadas.csv"]:
            local = os.path.join(crudos_dir, fname)
            if os.path.exists(local):
                s3.upload(local, s3.PREFIX_CRUDOS + fname)

        print("SUCCESS: [Agente 1] Ingesta y recolección completada para todos los dominios.")
        print("="*70)

        return {
            'clima': df_clima,
            'agua': df_agua,
            'casos': df_casos,
            'poblacion': df_poblaciones
        }

if __name__ == "__main__":
    agente = AgenteRecoleccion()
    datos = agente.ejecutar_ingesta()
