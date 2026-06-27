import os, sys, pandas as pd
sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
DB_DIR = os.path.join(BASE, 'Base de Datos', 'datos_procesados')

# Dataset maestro: solo quitar filas NIC
path_m = os.path.join(DB_DIR, 'dataset_maestro_mensual_latam.csv')
df_m   = pd.read_csv(path_m)
df_m   = df_m[df_m['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
df_m.to_csv(path_m, index=False)
s3.upload(path_m, s3.PREFIX_PROCESADOS + 'dataset_maestro_mensual_latam.csv')
print(f"Maestro: {len(df_m)} filas | paises: {sorted(df_m['pais'].unique())}")

# Dataset features: quitar filas NIC + columna pais_NIC
path_f = os.path.join(DB_DIR, 'dataset_features_latam.csv')
df_f   = pd.read_csv(path_f)
df_f   = df_f[df_f['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
if 'pais_NIC' in df_f.columns:
    df_f = df_f.drop(columns=['pais_NIC'])
    print("  Columna pais_NIC eliminada del features CSV")
df_f.to_csv(path_f, index=False)
s3.upload(path_f, s3.PREFIX_PROCESADOS + 'dataset_features_latam.csv')
print(f"Features: {len(df_f)} filas | columnas con 'pais_': {[c for c in df_f.columns if c.startswith('pais_')]}")
print("Listo.")
