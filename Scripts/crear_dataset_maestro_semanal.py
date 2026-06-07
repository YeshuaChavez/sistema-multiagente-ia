import pandas as pd
import numpy as np
import os
import sys
import requests
import time
import urllib.parse

sys.stdout.reconfigure(encoding='utf-8')

PAISES_10 = ['ARG', 'BOL', 'BRA', 'COL', 'DOM', 'ECU', 'MEX', 'NIC', 'PAN', 'PER']

# Coordenadas de contingencia (capitales/centros endémicos) en caso de que falle Nominatim
FALLBACK_COORDS = {
    "ARG": (-34.61, -58.38),
    "BOL": (-17.78, -63.18),
    "BRA": (-15.78, -47.93),
    "COL": (10.96, -74.79),
    "DOM": (18.48, -69.90),
    "ECU": (-2.17, -79.92),
    "MEX": (19.17, -96.13),
    "NIC": (12.13, -86.28),
    "PAN": (8.99, -79.52),
    "PER": (-5.19, -80.63),
}

# Nombres de países para geocodificación
COUNTRY_NAMES = {
    "ARG": "Argentina", "BOL": "Bolivia", "BRA": "Brazil", "COL": "Colombia",
    "DOM": "Dominican Republic", "ECU": "Ecuador", "MEX": "Mexico",
    "NIC": "Nicaragua", "PAN": "Panama", "PER": "Peru"
}

def geocode_department(iso, dept_name):
    """Obtiene latitud y longitud de un departamento usando Nominatim."""
    country_name = COUNTRY_NAMES.get(iso, "")
    query = f"{dept_name}, {country_name}"
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&limit=1"
    headers = {'User-Agent': 'DengueWeeklyPredictor/1.0 (yeshua.chavez@unmsm.edu.pe)'}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"  [Advertencia] Error al geocodificar '{query}': {e}")
    
    # Fallback si falla
    return FALLBACK_COORDS.get(iso, (0.0, 0.0))

def download_daily_weather(lat, lon, start_year, end_year):
    """Descarga clima diario desde NASA POWER para una coordenada."""
    url = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': f"{start_year}0101",
        'end': f"{end_year}1231",
        'format': 'JSON'
    }
    
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return data['properties']['parameter']
        except Exception as e:
            print(f"  [Reintento {attempt+1}] Error en descarga de clima: {e}")
            time.sleep(2)
            
    raise RuntimeError(f"No se pudo descargar datos de clima para lat={lat}, lon={lon}")

def main():
    print("=" * 65)
    print("  COMPILADOR DE DATASET MAESTRO SEMANAL SUBNACIONAL (LATAM)")
    print("=" * 65)
    
    # Directorio base
    base_dir = r"C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
    
    # Rutas
    dengue_path = os.path.join(base_dir, "Base de Datos", "Temporal_extract_V1_3.csv")
    population_path = os.path.join(base_dir, "Scripts", "poblacion_latam_2000_2024.csv")
    water_path = os.path.join(base_dir, "Scripts", "agua_latam_detallado.csv")
    coords_cache_path = os.path.join(base_dir, "Base de Datos", "departamentos_coordenadas.csv")
    output_path = os.path.join(base_dir, "Base de Datos", "dataset_maestro_semanal_latam.csv")
    
    # 1. Cargar Dengue Semanal
    print("1. Cargando datos epidemiológicos de dengue semanal...")
    df_dengue = pd.read_csv(dengue_path, usecols=['ISO_A0', 'adm_1_name', 'S_res', 'T_res', 'Year', 'dengue_total', 'calendar_start_date', 'calendar_end_date'])
    
    # Filtrar por los 10 países, semanal, años 2014-2024
    df_week = df_dengue[
        (df_dengue['ISO_A0'].isin(PAISES_10)) & 
        (df_dengue['T_res'] == 'Week') & 
        (df_dengue['Year'] >= 2014) & 
        (df_dengue['Year'] <= 2024)
    ].copy()
    
    # Rellenar departamentos nulos
    df_week['adm_1_name'] = df_week['adm_1_name'].fillna('UNKNOWN')
    
    # Unificar a nivel de departamento y semana (sumar casos si había subdivisión municipal Admin2)
    print("2. Agrupando casos epidemiológicos a nivel departamental (Admin1)...")
    df_cases = df_week.groupby(['ISO_A0', 'adm_1_name', 'Year', 'calendar_start_date', 'calendar_end_date'], as_index=False)['dengue_total'].sum()
    print(f"   Filas iniciales agrupadas: {len(df_cases)}")
    
    # 2. Geocodificación y Caché
    print("\n3. Resolviendo coordenadas de departamentos...")
    unique_depts = df_cases[['ISO_A0', 'adm_1_name']].drop_duplicates().reset_index(drop=True)
    print(f"   Total de departamentos a geocodificar: {len(unique_depts)}")
    
    # Cargar caché si existe
    if os.path.exists(coords_cache_path):
        print("   Cargando coordenadas guardadas en caché...")
        df_coords = pd.read_csv(coords_cache_path)
        unique_depts = pd.merge(unique_depts, df_coords, on=['ISO_A0', 'adm_1_name'], how='left')
    else:
        unique_depts['lat'] = np.nan
        unique_depts['lon'] = np.nan
        
    # Completar geocodificación de pendientes
    missing_coords = unique_depts[unique_depts['lat'].isnull()]
    if len(missing_coords) > 0:
        print(f"   Geocodificando {len(missing_coords)} departamentos faltantes...")
        resolved = []
        for idx, row in missing_coords.iterrows():
            iso = row['ISO_A0']
            dept = row['adm_1_name']
            print(f"   - Geocodificando: {dept} ({iso})...", end=" ", flush=True)
            lat, lon = geocode_department(iso, dept)
            unique_depts.loc[
                (unique_depts['ISO_A0'] == iso) & (unique_depts['adm_1_name'] == dept), 
                ['lat', 'lon']
            ] = [lat, lon]
            print(f"OK ({lat:.2f}, {lon:.2f})")
            time.sleep(1) # Respetar políticas de Nominatim
            
        # Guardar en caché
        unique_depts[['ISO_A0', 'adm_1_name', 'lat', 'lon']].to_csv(coords_cache_path, index=False)
        print("   Coordenadas guardadas en caché con éxito.")
        
    # 3. Descarga y procesamiento de clima por departamento
    print("\n4. Descargando y procesando clima semanal por departamento...")
    # Crear carpeta temporal de clima si no existe
    climate_cache_dir = os.path.join(base_dir, "Base de Datos", "clima_semanal_cache")
    os.makedirs(climate_cache_dir, exist_ok=True)
    
    climate_dfs = []
    
    for i, row in enumerate(unique_depts.itertuples(), 1):
        iso = row.ISO_A0
        dept = row.adm_1_name
        lat = row.lat
        lon = row.lon
        
        cache_fn = os.path.join(climate_cache_dir, f"{iso}_{dept.replace('/', '_').replace(' ', '_')}_clima.csv")
        
        # Filtramos las semanas correspondientes a este departamento
        df_dept_cases = df_cases[(df_cases['ISO_A0'] == iso) & (df_cases['adm_1_name'] == dept)].copy()
        if len(df_dept_cases) == 0:
            continue
            
        df_dept_cases['calendar_start_date'] = pd.to_datetime(df_dept_cases['calendar_start_date'])
        df_dept_cases['calendar_end_date'] = pd.to_datetime(df_dept_cases['calendar_end_date'])
        
        print(f"   [{i:03d}/{len(unique_depts)}] Clima para {dept} ({iso}) lat={lat:.2f}, lon={lon:.2f}...", end=" ", flush=True)
        
        df_weather = None
        if os.path.exists(cache_fn):
            df_weather = pd.read_csv(cache_fn)
            df_weather['calendar_start_date'] = pd.to_datetime(df_weather['calendar_start_date'])
            df_weather['calendar_end_date'] = pd.to_datetime(df_weather['calendar_end_date'])
            print("OK (Desde Caché)")
        else:
            try:
                # Descargar datos diarios
                daily_props = download_daily_weather(lat, lon, 2014, 2024)
                
                # Convertir a DataFrame
                records = []
                for date_str in daily_props['T2M_MAX'].keys():
                    records.append({
                        'date': pd.to_datetime(date_str, format='%Y%m%d'),
                        'tmax': daily_props['T2M_MAX'][date_str],
                        'tmin': daily_props['T2M_MIN'][date_str],
                        'precip': daily_props['PRECTOTCORR'][date_str],
                        'humidity': daily_props['RH2M'][date_str]
                    })
                df_daily = pd.DataFrame(records)
                df_daily.replace(-999.0, np.nan, inplace=True)
                df_daily = df_daily.ffill().bfill() # Imputar nulos locales
                
                # Mapear clima diario a semanas usando merge_asof
                df_weeks_map = df_dept_cases[['calendar_start_date', 'calendar_end_date']].drop_duplicates().sort_values('calendar_start_date')
                df_daily = df_daily.sort_values('date')
                
                df_mapped = pd.merge_asof(df_daily, df_weeks_map, left_on='date', right_on='calendar_start_date', direction='backward')
                # Filtrar que el día pertenezca al rango de la semana mapeada
                df_mapped = df_mapped[df_mapped['date'] <= df_mapped['calendar_end_date']]
                
                # Agregar semanalmente
                df_weekly_climate = df_mapped.groupby(['calendar_start_date', 'calendar_end_date'], as_index=False).agg({
                    'tmax': 'mean',
                    'tmin': 'mean',
                    'precip': 'sum',
                    'humidity': 'mean'
                })
                
                # Renombrar columnas
                df_weekly_climate.rename(columns={
                    'tmax': 'tmax_promedio',
                    'tmin': 'tmin_promedio',
                    'precip': 'precipitacion',
                    'humidity': 'humedad_promedio'
                }, inplace=True)
                
                # Guardar en caché local
                df_weekly_climate.to_csv(cache_fn, index=False)
                df_weather = df_weekly_climate
                print("OK (Descargado)")
                time.sleep(1) # Dormir 1 segundo para no sobrecargar NASA POWER
            except Exception as e:
                print(f"ERROR: {e}")
                # Si falla totalmente la descarga, usar la de contingencia de su capital
                continue
                
        if df_weather is not None and len(df_weather) > 0:
            # Calcular Lags climáticos por departamento
            df_weather = df_weather.sort_values('calendar_start_date').reset_index(drop=True)
            
            # Crear lags de semanas (1, 2, 3, 6)
            for var in ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']:
                for lag in [1, 2, 3, 6]:
                    col_name = f"{var.split('_')[0] if 'promedio' in var else var}_lag{lag}"
                    df_weather[col_name] = df_weather[var].shift(lag)
                    
            # Combinar con los casos semanales del departamento
            df_dept_merged = pd.merge(df_dept_cases, df_weather, on=['calendar_start_date', 'calendar_end_date'], how='inner')
            climate_dfs.append(df_dept_merged)

    if not climate_dfs:
        print("Error crítico: No se consolidó información climática.")
        return
        
    df_master = pd.concat(climate_dfs, ignore_index=True)
    
    # 4. Cargar Población y Agua JMP
    print("\n5. Fusionando con datos de población y cobertura de agua...")
    df_pop = pd.read_csv(population_path)
    df_water = pd.read_csv(water_path)
    
    # Unir población (Llaves: iso_a0 -> ISO_A0, ano -> Year)
    df_pop.rename(columns={'iso_a0': 'ISO_A0', 'ano': 'Year'}, inplace=True)
    df_master = pd.merge(df_master, df_pop[['ISO_A0', 'Year', 'poblacion']], on=['ISO_A0', 'Year'], how='inner')
    
    # Unir agua JMP
    df_water.rename(columns={'iso_a0': 'ISO_A0', 'ano': 'Year'}, inplace=True)
    # Quitar columna pais del agua para no tener redundancia
    if 'pais' in df_water.columns:
        df_water.drop(columns=['pais'], inplace=True)
    df_master = pd.merge(df_master, df_water, on=['ISO_A0', 'Year'], how='inner')
    
    # 5. Calcular Incidencia y Ajustar Nombres
    print("6. Normalizando la tasa de incidencia de dengue semanal...")
    df_master['incidencia_dengue'] = (df_master['dengue_total'] / df_master['poblacion']) * 100000
    df_master['incidencia_dengue'] = df_master['incidencia_dengue'].round(4)
    
    # Agregar columna con el nombre completo de país
    df_master['pais'] = df_master['ISO_A0'].map(COUNTRY_NAMES)
    
    # Renombrar columnas
    df_master.rename(columns={
        'ISO_A0': 'iso_a0',
        'dengue_total': 'casos_dengue',
        'Year': 'ano'
    }, inplace=True)
    
    # 5.5 Calcular Lags de Incidencia (Autorregresivos)
    print("6.5 Calculando variables autorregresivas (lags de incidencia)...")
    df_master = df_master.sort_values(by=['iso_a0', 'adm_1_name', 'calendar_start_date']).reset_index(drop=True)
    for lag in [1, 2, 3]:
        df_master[f'incidencia_lag{lag}'] = df_master.groupby(['iso_a0', 'adm_1_name'])['incidencia_dengue'].shift(lag)
        
    # Filtrar registros donde los lags (climáticos y de incidencia) son nulos (primeras semanas de 2014)
    cols_lags = [c for c in df_master.columns if 'lag' in c]
    df_master.dropna(subset=cols_lags, inplace=True)
    df_master.reset_index(drop=True, inplace=True)
    
    # Ordenar y guardar
    cols_base = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'calendar_start_date', 'calendar_end_date', 'casos_dengue', 'poblacion', 'incidencia_dengue']
    cols_water = ['agua_basica', 'agua_limitada', 'agua_no_mejorada', 'agua_superficial', 
                  'agua_entubada', 'agua_no_entubada', 'agua_disponible', 'agua_segura']
    cols_climate = ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']
    
    final_cols = cols_base + cols_water + cols_climate + cols_lags
    df_master = df_master[final_cols].copy()
    
    print(f"\n7. Guardando dataset maestro final en: {output_path}...")
    df_master.to_csv(output_path, index=False)
    
    print("\n" + "=" * 65)
    print(f"  Dataset Maestro Semanal Generado Exitosamente!")
    print(f"  Ruta del archivo     -> {output_path}")
    print(f"  Filas totales        -> {len(df_master):,}")
    print(f"  Columnas totales     -> {len(df_master.columns)}")
    print(f"  Países incluidos     -> {df_master['iso_a0'].nunique()}")
    print(f"  Departamentos/Edos   -> {df_master['adm_1_name'].nunique()}")
    print(f"  Valores nulos totales-> {df_master.isnull().sum().sum()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
