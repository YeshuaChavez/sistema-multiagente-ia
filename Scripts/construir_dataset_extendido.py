# -*- coding: utf-8 -*-
"""
Construye dataset extendido 2004-2019 para 7 países sólidos.
  - Dengue mensual: Temporal_extract_V1_3.csv (semanal → mensual)
  - Clima: NASA POWER mensual por departamento (descarga 2004-2009, usa existente 2010-2019)
  - Agua: backfill desde 2010 por departamento
  - Población: extrapolación lineal desde censos
  - Países: BOL, BRA, COL, MEX, NIC, PAN, PER  (sin ARG ni ECU)
  - Train: ≤ 2017  |  Test: 2018-2019
"""
import os, sys, json, time, requests
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
DB_DIR    = os.path.join(BASE_DIR, "Base de Datos")
RAW_DIR   = os.path.join(DB_DIR, "datos_crudos")
PROC_DIR  = os.path.join(DB_DIR, "datos_procesados")

PAISES_7  = ['BOL', 'BRA', 'COL', 'MEX', 'NIC', 'PAN', 'PER']
ANO_INICIO = 2004
ANO_FIN    = 2019

# ─────────────────────────────────────────────────────────────────────────────
# 1. DENGUE MENSUAL 2004-2019
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Procesando dengue mensual 2004-2019...")

df_raw = pd.read_csv(
    os.path.join(RAW_DIR, "Temporal_extract_V1_3.csv"),
    usecols=['ISO_A0','adm_0_name','adm_1_name','adm_2_name',
             'S_res','T_res','Year','dengue_total',
             'calendar_start_date','calendar_end_date'],
    low_memory=False
)

df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')
df_raw = df_raw[
    df_raw['ISO_A0'].isin(PAISES_7) &
    df_raw['Year'].between(ANO_INICIO, ANO_FIN) &
    df_raw['adm_1_name'].notna()
].copy()

# Extraer mes desde calendar_start_date
df_raw['calendar_start_date'] = pd.to_datetime(df_raw['calendar_start_date'], errors='coerce')
df_raw['mes'] = df_raw['calendar_start_date'].dt.month
# Para registros anuales sin fecha exacta, usar Year y distribución uniforme
df_raw.loc[df_raw['mes'].isna() & (df_raw['T_res']=='Year'), 'mes'] = 1

df_raw['dengue_total'] = pd.to_numeric(df_raw['dengue_total'], errors='coerce').fillna(0)

# Para Perú: Admin2 → Admin1 (adm_1_name ya contiene el departamento)
# Para el resto: Admin1 directamente
# Filtros de resolución por país
records = []
for iso in PAISES_7:
    sub = df_raw[df_raw['ISO_A0'] == iso].copy()
    if iso == 'PER':
        # Admin2 tiene adm_1_name = departamento
        sub = sub[sub['S_res'] == 'Admin2']
    else:
        # Preferir Admin1; si no existe, Admin2 con adm_1_name
        if 'Admin1' in sub['S_res'].values:
            sub = sub[sub['S_res'] == 'Admin1']
        else:
            sub = sub[sub['S_res'] == 'Admin2']
    records.append(sub)

df_dengue = pd.concat(records, ignore_index=True)

# Agregar a mensual: suma de casos por (iso, dept, año, mes)
df_dengue_m = (
    df_dengue
    .groupby(['ISO_A0','adm_0_name','adm_1_name','Year','mes'])['dengue_total']
    .sum()
    .reset_index()
)
df_dengue_m.columns = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue']
df_dengue_m['adm_1_name'] = df_dengue_m['adm_1_name'].str.strip().str.upper()
df_dengue_m['pais']       = df_dengue_m['pais'].str.title()

# Completar meses faltantes (semanal puede perder algún mes con cero casos)
combos = []
for (iso, pais, dept), grp in df_dengue_m.groupby(['iso_a0','pais','adm_1_name']):
    for ano in range(ANO_INICIO, ANO_FIN + 1):
        for mes in range(1, 13):
            val = grp[(grp['ano']==ano) & (grp['mes']==mes)]['casos_dengue'].sum()
            combos.append({'iso_a0': iso, 'pais': pais, 'adm_1_name': dept,
                           'ano': ano, 'mes': mes, 'casos_dengue': val})

df_dengue_m = pd.DataFrame(combos)
print(f"   Dengue: {len(df_dengue_m):,} filas  |  "
      f"{df_dengue_m['iso_a0'].nunique()} países  |  "
      f"{df_dengue_m['adm_1_name'].nunique()} departamentos")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLIMA NASA POWER — extender 2004-2009
# ─────────────────────────────────────────────────────────────────────────────
print("\n2. Clima: cargando existente (2010-2019) y descargando 2004-2009...")

clima_existente = pd.read_csv(os.path.join(RAW_DIR, "clima_nasa_crudo.csv"))
clima_existente = clima_existente[
    clima_existente['iso_a0'].isin(PAISES_7) &
    clima_existente['ano'].between(2010, ANO_FIN)
].copy()
clima_existente['adm_1_name'] = clima_existente['adm_1_name'].str.strip().str.upper()

coords = pd.read_csv(os.path.join(RAW_DIR, "departamentos_coordenadas.csv"))
coords = coords[coords['iso_a0'].isin(PAISES_7)].copy()
coords['adm_1_name'] = coords['adm_1_name'].str.strip().str.upper()

# Verificar qué departamentos ya tienen clima 2004-2009
depts_con_clima_anterior = set(
    zip(clima_existente[clima_existente['ano'] < 2010]['iso_a0'],
        clima_existente[clima_existente['ano'] < 2010]['adm_1_name'])
)

NASA_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

def descargar_clima_dept(lat, lon, start_yr=2004, end_yr=2009):
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
        "community": "RE",
        "longitude": lon, "latitude": lat,
        "start": str(start_yr), "end": str(end_yr),
        "format": "JSON",
    }
    for intento in range(3):
        try:
            r = requests.get(NASA_URL, params=params, timeout=90)
            r.raise_for_status()
            props = r.json()["properties"]["parameter"]
            filas = []
            for clave, tmax in props["T2M_MAX"].items():
                if clave.endswith("13"):
                    continue
                filas.append({
                    "ano": int(clave[:4]), "mes": int(clave[4:]),
                    "tmax_promedio":    tmax,
                    "tmin_promedio":    props["T2M_MIN"][clave],
                    "precipitacion":    props["PRECTOTCORR"][clave],
                    "humedad_promedio": props["RH2M"][clave],
                })
            return filas
        except Exception as e:
            print(f"      [reintento {intento+1}] {e}")
            time.sleep(3)
    return []

nuevos_clima = []
total_depts  = len(coords)
cache_path   = os.path.join(BASE_DIR, "Scripts", "_clima_2004_2009_cache.json")

# Cargar cache si existe (para no re-descargar si se interrumpe)
cache = {}
if os.path.exists(cache_path):
    with open(cache_path) as f:
        cache = json.load(f)
    print(f"   Cache encontrado: {len(cache)} departamentos ya descargados")

for i, row in coords.iterrows():
    iso, dept, lat, lon = row['iso_a0'], row['adm_1_name'], row['lat'], row['lon']
    key = f"{iso}|{dept}"

    if key in cache:
        nuevos_clima.extend(cache[key])
        continue

    print(f"   [{i+1:3d}/{total_depts}] {iso} - {dept:<30}", end=" ", flush=True)
    filas = descargar_clima_dept(lat, lon, 2004, 2009)
    if filas:
        for f in filas:
            f.update({'iso_a0': iso, 'adm_1_name': dept})
        nuevos_clima.extend(filas)
        cache[key] = filas
        print(f"OK ({len(filas)} meses)")
    else:
        print("FALLO — usando interpolación")
    time.sleep(0.8)

# Guardar cache
with open(cache_path, "w") as f:
    json.dump(cache, f)

df_clima_nueva = pd.DataFrame(nuevos_clima)
df_clima_nueva = df_clima_nueva[df_clima_nueva['ano'].between(ANO_INICIO, 2009)]
df_clima_nueva.replace(-999.0, float("nan"), inplace=True)

# Añadir columna pais desde coords
pais_map = dict(zip(coords['iso_a0'], coords['iso_a0']))  # placeholder; se usa iso
df_clima = pd.concat([df_clima_nueva, clima_existente], ignore_index=True)
df_clima  = df_clima.drop_duplicates(['iso_a0','adm_1_name','ano','mes']).sort_values(
    ['iso_a0','adm_1_name','ano','mes']).reset_index(drop=True)

print(f"   Clima total: {len(df_clima):,} filas (2004-2019, 7 países)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. AGUA BÁSICA — backfill desde 2010
# ─────────────────────────────────────────────────────────────────────────────
print("\n3. Extendiendo agua_basica hacia atrás (backfill desde 2010)...")

agua_orig = pd.read_csv(os.path.join(RAW_DIR, "agua_jmp_crudo.csv"))
agua_orig = agua_orig[agua_orig['iso_a0'].isin(PAISES_7)].copy()
agua_orig['adm_1_name'] = agua_orig['adm_1_name'].str.strip().str.upper()

agua_ext = []
for (iso, dept), grp in agua_orig.groupby(['iso_a0','adm_1_name']):
    # Valor de 2010 (el más antiguo disponible)
    val_2010 = grp[grp['ano']==2010]['agua_basica'].mean()
    if pd.isna(val_2010):
        val_2010 = grp['agua_basica'].mean()
    for ano in range(ANO_INICIO, 2010):
        for mes in range(1, 13):
            agua_ext.append({'iso_a0': iso, 'adm_1_name': dept,
                             'ano': ano, 'mes': mes, 'agua_basica': val_2010})

df_agua = pd.concat([pd.DataFrame(agua_ext), agua_orig], ignore_index=True)
df_agua  = df_agua[df_agua['ano'].between(ANO_INICIO, ANO_FIN)]
df_agua  = df_agua.drop_duplicates(['iso_a0','adm_1_name','ano','mes'])
print(f"   Agua: {len(df_agua):,} filas")

# ─────────────────────────────────────────────────────────────────────────────
# 4. POBLACIÓN — extrapolación lineal desde censos
# ─────────────────────────────────────────────────────────────────────────────
print("\n4. Generando población 2004-2019 por extrapolación de censos...")

pob_dir   = os.path.join(RAW_DIR, "poblacion")
pais_names = {
    'BOL': 'bolivia', 'BRA': 'brazil', 'COL': 'colombia',
    'MEX': 'mexico',  'NIC': 'nicaragua', 'PAN': 'panama', 'PER': 'peru'
}

def interpolar(t, t_arr, p_arr):
    t_arr, p_arr = np.array(t_arr), np.array(p_arr)
    idx = np.argsort(t_arr)
    t_arr, p_arr = t_arr[idx], p_arr[idx]
    if t in t_arr:
        return int(p_arr[t_arr == t][0])
    for i in range(len(t_arr)-1):
        if t_arr[i] < t < t_arr[i+1]:
            return int(round(p_arr[i] + (p_arr[i+1]-p_arr[i])/(t_arr[i+1]-t_arr[i])*(t-t_arr[i])))
    if t < t_arr[0]:
        slope = (p_arr[1]-p_arr[0])/(t_arr[1]-t_arr[0]) if len(t_arr)>1 else 0
        return max(1, int(round(p_arr[0] + slope*(t-t_arr[0]))))
    slope = (p_arr[-1]-p_arr[-2])/(t_arr[-1]-t_arr[-2]) if len(t_arr)>1 else 0
    return max(1, int(round(p_arr[-1] + slope*(t-t_arr[-1]))))

pob_rows = []
for iso, pais_fname in pais_names.items():
    fpath = os.path.join(pob_dir, f"censos_crudos_{pais_fname}.csv")
    if not os.path.exists(fpath):
        print(f"   [AVISO] No se encontró censo para {iso}")
        continue
    df_censo = pd.read_csv(fpath)
    df_censo.columns = df_censo.columns.str.lower()
    # Normalizar columnas
    col_dept = [c for c in df_censo.columns if 'adm' in c or 'dept' in c or 'region' in c or 'name' in c][0] \
               if any('adm' in c or 'dept' in c or 'region' in c or 'name' in c for c in df_censo.columns) \
               else df_censo.columns[1]
    col_year = [c for c in df_censo.columns if 'ano' in c or 'year' in c or 'anio' in c][0]
    col_pob  = [c for c in df_censo.columns if 'pob' in c or 'pop' in c][0]

    for dept, grp in df_censo.groupby(col_dept):
        t_list = grp[col_year].tolist()
        p_list = grp[col_pob].tolist()
        dept_u = str(dept).strip().upper()
        for ano in range(ANO_INICIO, ANO_FIN+1):
            pob_rows.append({
                'iso_a0': iso, 'adm_1_name': dept_u,
                'ano': ano, 'poblacion': interpolar(ano, t_list, p_list)
            })

df_pob = pd.DataFrame(pob_rows)
print(f"   Población: {len(df_pob):,} filas  |  {df_pob['adm_1_name'].nunique()} departamentos")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MERGE — construir dataset maestro
# ─────────────────────────────────────────────────────────────────────────────
print("\n5. Construyendo dataset maestro...")

# Normalizar nombres
for df in [df_dengue_m, df_clima, df_agua, df_pob]:
    df['adm_1_name'] = df['adm_1_name'].str.strip().str.upper()
    df['iso_a0']     = df['iso_a0'].str.strip().str.upper()

# Merge dengue + población (por iso, dept, año)
df_m = pd.merge(df_dengue_m, df_pob, on=['iso_a0','adm_1_name','ano'], how='inner')

# Merge + clima (por iso, dept, año, mes)
df_clima_sel = df_clima[['iso_a0','adm_1_name','ano','mes',
                          'tmax_promedio','tmin_promedio','precipitacion','humedad_promedio']]
df_m = pd.merge(df_m, df_clima_sel, on=['iso_a0','adm_1_name','ano','mes'], how='inner')

# Merge + agua (por iso, dept, año, mes)
df_agua_sel = df_agua[['iso_a0','adm_1_name','ano','mes','agua_basica']]
df_m = pd.merge(df_m, df_agua_sel, on=['iso_a0','adm_1_name','ano','mes'], how='left')
df_m['agua_basica'] = df_m['agua_basica'].fillna(df_m.groupby(['iso_a0','adm_1_name'])['agua_basica'].transform('mean'))

# Densidad poblacional
areas = pd.read_csv(os.path.join(RAW_DIR, "departamentos_areas.csv"))
areas['adm_1_name'] = areas['adm_1_name'].str.strip().str.upper()
areas['iso_a0']     = areas['iso_a0'].str.strip().str.upper()
df_m = pd.merge(df_m, areas[['iso_a0','adm_1_name','area_km2']], on=['iso_a0','adm_1_name'], how='left')
df_m['densidad_poblacion'] = (df_m['poblacion'] / df_m['area_km2'].replace(0, np.nan)).fillna(0)
df_m.drop(columns=['area_km2'], inplace=True, errors='ignore')

# Incidencia
df_m['incidencia_dengue'] = (df_m['casos_dengue'] / df_m['poblacion'].replace(0, np.nan) * 100000).round(4).fillna(0)

# Filtrar departamentos activos (al menos 100 casos anuales en algún año)
yearly = df_m.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df_m = df_m[yearly > 100].reset_index(drop=True)

df_m = df_m.sort_values(['iso_a0','adm_1_name','ano','mes']).reset_index(drop=True)

output_path = os.path.join(PROC_DIR, "dataset_maestro_mensual_latam_v2.csv")
df_m.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"  Dataset guardado: {output_path}")
print(f"  Filas:       {len(df_m):,}")
print(f"  Países:      {df_m['iso_a0'].nunique()} — {sorted(df_m['iso_a0'].unique())}")
print(f"  Años:        {df_m['ano'].min()} – {df_m['ano'].max()}")
print(f"  Departamentos: {df_m['adm_1_name'].nunique()}")
print(f"\n  Train ≤ 2017: {(df_m['ano']<=2017).sum():,} filas")
print(f"  Test 2018-19: {df_m['ano'].between(2018,2019).sum():,} filas")
print(f"{'='*60}")
