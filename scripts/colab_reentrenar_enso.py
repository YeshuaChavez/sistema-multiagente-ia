# -*- coding: utf-8 -*-
"""
NOTEBOOK COLAB — Reentrenamiento XGBoost con índices ENSO continuos
====================================================================
Instrucciones:
1. Sube dataset_features_latam_enso.csv a tu Google Drive
2. Copia este script a un notebook de Colab
3. Monta Drive y ajusta DATASET_PATH
4. Ejecuta celda por celda

El nuevo modelo reemplaza indicador_nino/indicador_nina (binarios)
por nino34_index, nino12_index y sus lags (continuos).
"""

# ── CELDA 1: Instalar dependencias ────────────────────────────────────────────
# !pip install xgboost==2.1.0 shap optuna scikit-learn pandas numpy boto3 -q

# ── CELDA 2: Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import json, pickle, os, boto3
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ── CELDA 3: Configuración ────────────────────────────────────────────────────
# Ajusta esta ruta a donde subiste el CSV en Drive
DATASET_PATH = "/content/drive/MyDrive/dengue/dataset_features_latam_enso.csv"
OUT_DIR      = "/content/modelos_enso"
SEMILLA      = 42
N_TRIALS     = 50          # Optuna trials (sube a 100 para mejor resultado)
os.makedirs(OUT_DIR, exist_ok=True)

# ── CELDA 4: Cargar y preparar datos ─────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {df.shape}")

# Filtrar países con datos suficientes
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
print(f"Después de filtro: {df.shape}")

# Excluir columnas no-feature
COLS_EXCLUIR = [
    'iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
    'casos_dengue', 'poblacion', 'incidencia_dengue',
    # Reemplazamos los binarios por los continuos → excluir binarios
    'indicador_nino', 'indicador_nina',
]
COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
print(f"Features ({len(COLS_FEAT)}): {COLS_FEAT}")

# Split cronológico (últimos 2 años = test)
max_ano   = int(df['ano'].max())
split_ano = max_ano - 2
df_train  = df[df['ano'] <= split_ano].copy()
df_test   = df[df['ano'] >  split_ano].copy()
print(f"Train: {len(df_train)} | Test: {len(df_test)} | Split: ≤{split_ano} / >{split_ano}")

X_train = df_train[COLS_FEAT]
y_train = df_train['incidencia_dengue']
X_test  = df_test[COLS_FEAT]
y_test  = df_test['incidencia_dengue']
y_train_log = np.log1p(y_train)

# ── CELDA 5: Baseline ─────────────────────────────────────────────────────────
pipe_base = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBRegressor(random_state=SEMILLA, n_jobs=-1, verbosity=0))
])
pipe_base.fit(X_train, y_train_log)
pred_base = np.expm1(pipe_base.predict(X_test))
r2_base   = r2_score(np.log1p(y_test), np.log1p(pred_base))
mae_base  = mean_absolute_error(y_test, pred_base)
print(f"Baseline — R²(log)={r2_base*100:.2f}%  MAE={mae_base:.4f}")

# ── CELDA 6: Optimización Bayesiana (Optuna) ──────────────────────────────────
anos_train = df_train['ano'].values
anos_unicos = sorted(df_train['ano'].unique())

def make_folds(anos_unicos, n_folds=5):
    """Folds cronológicos: siempre entrena en pasado, valida en futuro."""
    folds = []
    step  = max(1, len(anos_unicos) // (n_folds + 1))
    for i in range(1, n_folds + 1):
        cutoff = anos_unicos[min(i * step, len(anos_unicos) - 1)]
        idx_tr = np.where(anos_train <= cutoff)[0]
        idx_va = np.where(anos_train == cutoff + 1)[0] if cutoff + 1 in anos_unicos else np.array([])
        if len(idx_tr) > 0 and len(idx_va) > 0:
            folds.append((idx_tr, idx_va))
    return folds

FOLDS = make_folds(anos_unicos)
print(f"Folds cronológicos: {len(FOLDS)}")

X_train_arr = X_train.values
y_train_arr = y_train_log.values

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "objective": "reg:squarederror",
        "random_state": SEMILLA, "verbosity": 0, "n_jobs": -1,
    }
    scores = []
    for idx_tr, idx_va in FOLDS:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  XGBRegressor(**params))
        ])
        pipe.fit(X_train_arr[idx_tr], y_train_arr[idx_tr])
        pred = pipe.predict(X_train_arr[idx_va])
        scores.append(r2_score(y_train_arr[idx_va], pred))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"Mejor R² CV: {study.best_value*100:.2f}%")
print(f"Mejores params: {study.best_params}")

# ── CELDA 7: Entrenar modelo final con mejores parámetros ─────────────────────
best_params = {
    **study.best_params,
    "objective": "reg:squarederror",
    "random_state": SEMILLA, "verbosity": 0, "n_jobs": -1,
}
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBRegressor(**best_params))
])
pipeline.fit(X_train, y_train_log)

pred_log = pipeline.predict(X_test)
pred     = np.expm1(pred_log)
r2_log   = r2_score(np.log1p(y_test), pred_log)
mae      = mean_absolute_error(y_test, pred)
print(f"Final — R²(log)={r2_log*100:.2f}%  MAE={mae:.4f}")
print(f"Mejora sobre baseline: R² {(r2_log-r2_base)*100:+.2f}pp  MAE {mae-mae_base:+.4f}")

# ── CELDA 8: SHAP con mean(|SHAP|) ────────────────────────────────────────────
print("Calculando SHAP...")
modelo    = pipeline.named_steps['model']
X_test_sc = pipeline[:-1].transform(X_test)
explainer = shap.TreeExplainer(modelo)
shap_vals = explainer.shap_values(X_test_sc)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_dict = dict(sorted(
    {f: float(v) for f, v in zip(COLS_FEAT, mean_abs_shap)}.items(),
    key=lambda x: x[1], reverse=True
))
print("Top 10 SHAP:")
for i, (k, v) in enumerate(list(shap_dict.items())[:10], 1):
    print(f"  #{i:02d}  {k:<30} {v:.4f}")

# ── CELDA 9: Guardar artefactos ───────────────────────────────────────────────
with open(f"{OUT_DIR}/pipeline_ml.pkl",    "wb") as f: pickle.dump(pipeline, f)
with open(f"{OUT_DIR}/cols_feat.pkl",      "wb") as f: pickle.dump(COLS_FEAT, f)
with open(f"{OUT_DIR}/shap_importance.json","w") as f: json.dump(shap_dict, f, indent=4)

metrics = {
    "r2_log": round(float(r2_log), 4),
    "mae":    round(float(mae), 4),
    "r2_baseline": round(float(r2_base), 4),
    "n_features": len(COLS_FEAT),
    "split_ano": split_ano,
    "best_params": study.best_params,
}
with open(f"{OUT_DIR}/xgb_config.json", "w") as f: json.dump(metrics, f, indent=4)
print(f"Artefactos guardados en {OUT_DIR}/")

# ── CELDA 10 (OPCIONAL): Subir a S3 ──────────────────────────────────────────
# Configura tus credenciales AWS antes de ejecutar esta celda
# import os
# os.environ["AWS_ACCESS_KEY_ID"]     = "TU_KEY"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "TU_SECRET"
# os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"
#
# s3 = boto3.client("s3")
# BUCKET = "epipredict-dengue"
# for fname in ["pipeline_ml.pkl", "cols_feat.pkl", "shap_importance.json", "xgb_config.json"]:
#     s3.upload_file(f"{OUT_DIR}/{fname}", BUCKET, f"modelos/{fname}")
#     print(f"Subido: {fname}")
