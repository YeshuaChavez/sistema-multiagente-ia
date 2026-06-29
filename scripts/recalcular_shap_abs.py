# -*- coding: utf-8 -*-
"""
Recomputa shap_importance.json usando mean(|SHAP|) en lugar del mean(SHAP)
firmado que se guardó originalmente.
Usa el modelo y el pipeline ya entrenados — no requiere reentrenamiento.
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
import shap

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, "data", "models")
DATA  = os.path.join(BASE, "data", "processed", "dataset_features_latam.csv")

print("[SHAP-abs] Cargando artefactos...")
with open(os.path.join(MODEL, "pipeline_ml.pkl"), "rb") as f:
    pipeline = pickle.load(f)
with open(os.path.join(MODEL, "cols_feat.pkl"), "rb") as f:
    COLS_FEAT = pickle.load(f)

print(f"[SHAP-abs] Features: {len(COLS_FEAT)}")

print("[SHAP-abs] Leyendo dataset...")
df = pd.read_csv(DATA)
print(f"[SHAP-abs] Dataset: {df.shape}")

COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
                'casos_dengue', 'poblacion', 'incidencia_dengue']

max_ano   = int(df['ano'].max())
split_ano = max_ano - 2
df_test   = df[df['ano'] > split_ano].copy()
print(f"[SHAP-abs] Test set (anos >{split_ano}): {df_test.shape}")

X_test_raw = df_test[COLS_FEAT]

print("[SHAP-abs] Transformando test set (imputa + escala)...")
step_names = list(pipeline.named_steps.keys())
print(f"[SHAP-abs] Pipeline steps: {step_names}")
preproc_steps = step_names[:-1]  # todos menos el último (el modelo)
X_test_sc = pipeline[:-1].transform(X_test_raw)

modelo = pipeline.named_steps[step_names[-1]]
print("[SHAP-abs] Calculando TreeSHAP (esto puede tardar 1-2 min)...")
explainer = shap.TreeExplainer(modelo)
shap_vals = explainer.shap_values(X_test_sc)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]

# mean(|SHAP|) — magnitud absoluta promedio, siempre positiva
mean_abs_shap = np.abs(shap_vals).mean(axis=0)

shap_dict = dict(sorted(
    {f: float(v) for f, v in zip(COLS_FEAT, mean_abs_shap)}.items(),
    key=lambda x: x[1], reverse=True
))

out_path = os.path.join(MODEL, "shap_importance.json")
with open(out_path, "w") as f:
    json.dump(shap_dict, f, indent=4)

print(f"[SHAP-abs] Guardado en {out_path}")
print("[SHAP-abs] Top 10:")
for i, (feat, val) in enumerate(list(shap_dict.items())[:10], 1):
    print(f"  #{i:02d}  {feat:<35} {val:.4f}")

print("\n[SHAP-abs] Intentando subir a S3...")
try:
    import sys
    sys.path.insert(0, os.path.join(BASE, "agents"))
    import s3_client as s3
    s3.upload(out_path, s3.PREFIX_MODELOS + "shap_importance.json")
    print("[SHAP-abs] Subido a S3 correctamente. El backend cargará los nuevos valores al próximo reinicio.")
except Exception as e:
    print(f"[SHAP-abs] No se pudo subir a S3: {e}")
    print(f"[SHAP-abs] Sube manualmente: aws s3 cp {out_path} s3://epipredict-dengue/modelos/shap_importance.json")
