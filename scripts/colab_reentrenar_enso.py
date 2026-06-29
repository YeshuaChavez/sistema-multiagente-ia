# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 1 — Instalar dependencias                                ║
# ╚══════════════════════════════════════════════════════════════════╝
# !pip install xgboost shap optuna scikit-learn -q


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 2 — Subir el CSV desde tu PC                             ║
# ╚══════════════════════════════════════════════════════════════════╝
from google.colab import files
uploaded = files.upload()   # selecciona dataset_features_latam_enso.csv
DATASET_PATH = list(uploaded.keys())[0]
print(f"Archivo cargado: {DATASET_PATH}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 3 — Imports                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
import pandas as pd
import numpy as np
import json, pickle, os
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 4 — Cargar datos y definir features                      ║
# ╚══════════════════════════════════════════════════════════════════╝
df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {df.shape}")

# Filtrar países con datos suficientes
yearly = df.groupby(['pais', 'ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
print(f"Después de filtro: {df.shape}")

# Excluir columnas no-feature
# IMPORTANTE: excluimos los binarios — usamos los índices continuos
COLS_EXCLUIR = [
    'iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
    'casos_dengue', 'poblacion', 'incidencia_dengue',
    'indicador_nino', 'indicador_nina',   # reemplazados por nino34_index / nino12_index
]
COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
print(f"\nFeatures totales: {len(COLS_FEAT)}")

# Verificar que los índices ENSO están incluidos
enso_cols = [c for c in COLS_FEAT if 'nino' in c]
print(f"Columnas ENSO nuevas: {enso_cols}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 5 — Split cronológico                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
max_ano   = int(df['ano'].max())
split_ano = max_ano - 2
df_train  = df[df['ano'] <= split_ano].copy()
df_test   = df[df['ano'] >  split_ano].copy()
print(f"Train: {len(df_train)} registros (≤{split_ano})")
print(f"Test:  {len(df_test)} registros  (>{split_ano})")

X_train = df_train[COLS_FEAT]
y_train = df_train['incidencia_dengue']
X_test  = df_test[COLS_FEAT]
y_test  = df_test['incidencia_dengue']
y_train_log = np.log1p(y_train)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 6 — Baseline (sin optimizar)                             ║
# ╚══════════════════════════════════════════════════════════════════╝
pipe_base = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))
])
pipe_base.fit(X_train, y_train_log)
pred_base = np.expm1(pipe_base.predict(X_test))
r2_base   = r2_score(np.log1p(y_test), np.log1p(pred_base))
mae_base  = mean_absolute_error(y_test, pred_base)
print(f"Baseline — R²(log): {r2_base*100:.2f}%  |  MAE: {mae_base:.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 7 — Optimización Bayesiana con Optuna                    ║
# ╚══════════════════════════════════════════════════════════════════╝
N_TRIALS   = 50   # sube a 100 si quieres mejor resultado y tienes tiempo
anos_train = df_train['ano'].values
anos_unicos = sorted(df_train['ano'].unique())

# Folds cronológicos
folds = []
step  = max(1, len(anos_unicos) // 6)
for i in range(1, 6):
    cutoff  = anos_unicos[min(i * step, len(anos_unicos) - 1)]
    idx_tr  = np.where(anos_train <= cutoff)[0]
    next_yr = cutoff + 1
    idx_va  = np.where(anos_train == next_yr)[0] if next_yr in anos_unicos else np.array([])
    if len(idx_tr) > 0 and len(idx_va) > 0:
        folds.append((idx_tr, idx_va))
print(f"Folds cronológicos: {len(folds)}")

X_arr = X_train.values
y_arr = y_train_log.values

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
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
    }
    scores = []
    for idx_tr, idx_va in folds:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(**params))])
        pipe.fit(X_arr[idx_tr], y_arr[idx_tr])
        scores.append(r2_score(y_arr[idx_va], pipe.predict(X_arr[idx_va])))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\nMejor R² CV: {study.best_value*100:.2f}%")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 8 — Entrenar modelo final y evaluar                      ║
# ╚══════════════════════════════════════════════════════════════════╝
best_params = {**study.best_params, "objective": "reg:squarederror",
               "random_state": 42, "verbosity": 0, "n_jobs": -1}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBRegressor(**best_params))
])
pipeline.fit(X_train, y_train_log)

pred_log = pipeline.predict(X_test)
pred     = np.expm1(pred_log)
r2_log   = r2_score(np.log1p(y_test), pred_log)
mae      = mean_absolute_error(y_test, pred)
r2_lineal = r2_score(y_test, pred)

print("=" * 50)
print("        MÉTRICAS FINALES — ENSO CONTINUO")
print("=" * 50)
print(f"  R² logarítmico:  {r2_log*100:.2f}%  (antes: {r2_base*100:.2f}%  | Δ {(r2_log-r2_base)*100:+.2f}pp)")
print(f"  MAE:             {mae:.4f}         (antes: {mae_base:.4f}       | Δ {mae-mae_base:+.4f})")
print(f"  R² lineal:       {r2_lineal*100:.2f}%")
print("=" * 50)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 9 — SHAP global con mean(|SHAP|)                         ║
# ╚══════════════════════════════════════════════════════════════════╝
print("Calculando SHAP (puede tardar 1-2 min)...")
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

print("\nTop 15 features por importancia SHAP:")
for i, (k, v) in enumerate(list(shap_dict.items())[:15], 1):
    marker = " ← ENSO nuevo" if 'nino' in k else ""
    print(f"  #{i:02d}  {k:<35} {v:.4f}{marker}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELDA 10 — Guardar artefactos para descargar                   ║
# ╚══════════════════════════════════════════════════════════════════╝
os.makedirs("/content/modelos_enso", exist_ok=True)

with open("/content/modelos_enso/pipeline_ml.pkl",     "wb") as f: pickle.dump(pipeline, f)
with open("/content/modelos_enso/cols_feat.pkl",        "wb") as f: pickle.dump(COLS_FEAT, f)
with open("/content/modelos_enso/shap_importance.json", "w") as f: json.dump(shap_dict, f, indent=4)

metrics_out = {
    "r2_log":      round(float(r2_log), 4),
    "r2_lineal":   round(float(r2_lineal), 4),
    "mae":         round(float(mae), 4),
    "r2_baseline": round(float(r2_base), 4),
    "n_features":  len(COLS_FEAT),
    "enso_features": enso_cols,
    "best_params": study.best_params,
}
with open("/content/modelos_enso/metrics.json", "w") as f: json.dump(metrics_out, f, indent=4)

print("Artefactos guardados en /content/modelos_enso/")
print("Descarga desde el panel de archivos de Colab si las métricas mejoran.")
