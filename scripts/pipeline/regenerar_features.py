"""Regenera dataset_features_latam.csv con las 3 nuevas features de tendencia."""
import os, sys
sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import pandas as pd, numpy as np
import s3_client as s3
from agente_2_preprocesamiento import AgentePreprocesamiento

BASE   = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
ag2    = AgentePreprocesamiento(base_dir=BASE)

# Leer dataset base ya limpio (sin NIC)
df_base = pd.read_csv(ag2.output_base)
print(f"Base: {len(df_base)} filas | paises: {sorted(df_base['pais'].unique())}")

df_feat = ag2.generar_features(df_base)
df_feat.to_csv(ag2.output_feat, index=False)
s3.upload(ag2.output_feat, s3.PREFIX_PROCESADOS + 'dataset_features_latam.csv')
nuevas = [c for c in df_feat.columns if c in ('tendencia_1m','tendencia_3m','fase_ascendente')]
print(f"Features: {df_feat.shape[1]} columnas | nuevas: {nuevas}")
print("Listo.")
