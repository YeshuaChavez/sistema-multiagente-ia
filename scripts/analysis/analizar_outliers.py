import sys, os
import numpy as np, pandas as pd

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)

# Estadisticas por departamento
stats = df.groupby(['pais','adm_1_name'])['incidencia_dengue'].agg(
    media='mean', mediana='median', std='std', maximo='max', p95=lambda x: x.quantile(0.95)
).reset_index()

stats['cv'] = stats['std'] / (stats['media'] + 1)  # coef variacion

print("=== Top 20 departamentos por MAXIMO de incidencia ===")
print(stats.nlargest(20, 'maximo')[['pais','adm_1_name','media','mediana','maximo','p95']].to_string(index=False))

print("\n=== Top 20 departamentos por MEDIA de incidencia ===")
print(stats.nlargest(20, 'media')[['pais','adm_1_name','media','mediana','maximo','p95']].to_string(index=False))

print("\n=== Top 20 por COEF. DE VARIACION (mas erraticos) ===")
print(stats.nlargest(20, 'cv')[['pais','adm_1_name','media','mediana','maximo','cv']].to_string(index=False))

# Distribucion global
print(f"\n=== Distribucion global de incidencia_dengue ===")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"  P{p:5.1f}: {df['incidencia_dengue'].quantile(p/100):.1f}")
print(f"  Max   : {df['incidencia_dengue'].max():.1f}")
print(f"  Total filas: {len(df)}")

# Cuantos registros y departamentos sobre distintos umbrales
for umbral in [500, 1000, 2000, 5000]:
    n_filas = (df['incidencia_dengue'] > umbral).sum()
    deptos  = df[df['incidencia_dengue'] > umbral][['pais','adm_1_name']].drop_duplicates()
    print(f"\n  > {umbral:5d}: {n_filas} filas ({n_filas/len(df)*100:.2f}%) | {len(deptos)} departamentos")
    if len(deptos) <= 10:
        for _, r in deptos.iterrows():
            print(f"           {r['pais']:12s} {r['adm_1_name']}")
