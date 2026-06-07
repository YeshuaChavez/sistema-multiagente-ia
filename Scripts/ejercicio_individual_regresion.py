# -*- coding: utf-8 -*-
"""
Sistema de Machine Learning para la Regresión Supervisada de Incidencia de Dengue
Dataset: dataset_maestro_latam.csv (Unificado del Proyecto)
Institución: Universidad Nacional Mayor de San Marcos (UNMSM)
Curso: Aprendizaje Supervisado / Inteligencia Artificial

Modelos implementados:
- Caja Blanca: Regresión Ridge (Lineal Regularizada)
- Caja Negra 1: Random Forest Regressor
- Caja Negra 2: XGBoost Regressor
"""

# =====================================================================
# (1) IMPORTACIONES
# =====================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración estética de los gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# =====================================================================
# (2) CARGA Y EXPLORACIÓN BÁSICA DEL DATASET MAESTRO
# =====================================================================
print("=== (2) CARGA Y EXPLORACIÓN DEL DATASET ===")
DATASET_PATH = "Base de Datos/dataset_maestro_latam.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró el dataset maestro en: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Dimensiones del dataset maestro: {df.shape[0]} filas, {df.shape[1]} columnas.")
print("\nPrimeras 3 filas del dataset:")
print(df.head(3))

# =====================================================================
# (3) PREPROCESAMIENTO DE VARIABLES
# =====================================================================
print("\n=== (3) PREPROCESAMIENTO DE VARIABLES ===")

# Definimos las columnas que NO son características predictoras
cols_to_drop = ['iso_a0', 'pais', 'ano', 'mes', 'casos_dengue', 'poblacion', 'incidencia_dengue']

# Separar características (X) y variable objetivo (y)
# Predictoras: variables de agua, clima actual y lags de clima
X = df.drop(columns=cols_to_drop)
y = df['incidencia_dengue']

print(f"Cantidad de características predictoras (X): {X.shape[1]}")
print(f"Variable objetivo (y): 'incidencia_dengue' (Tasa de casos por 100k habitantes)")

# Imputación de nulos por si acaso existieran
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Estandarización de las características predictoras
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print("Preprocesamiento completado:")
print("  - Exclusión de variables identificadoras e intermedias.")
print("  - Imputación de seguridad y escalado estándar de características predictoras.")

# =====================================================================
# (4) PARTICIÓN EN ENTRENAMIENTO (80%) Y PRUEBA (20%)
# =====================================================================
print("\n=== (4) PARTICIÓN 80/20 ===")
# Hacemos la división aleatoria estratificada o estándar (estándar para regresión simple)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} registros.")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} registros.")

# =====================================================================
# (5) MODELO DE CAJA BLANCA CON VALIDACIÓN CRUZADA K=5 (DEFECTO)
# =====================================================================
print("\n=== (5) MODELO CAJA BLANCA (RIDGE REGRESSION) ===")
# Instanciamos Ridge Regression con hiperparámetros por defecto
model_ridge = Ridge()
print("Modelo instanciado con parámetros por defecto.")

# =====================================================================
# (6) MODELOS DE CAJA NEGRA CON VALIDACIÓN CRUZADA K=5 (DEFECTO)
# =====================================================================
print("\n=== (6) MODELOS CAJA NEGRA (RANDOM FOREST & XGBOOST) ===")
# Instanciamos Random Forest Regressor con hiperparámetros por defecto
model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# Instanciamos XGBoost Regressor con hiperparámetros por defecto
model_xgb = XGBRegressor(random_state=42, n_jobs=-1)
print("Modelos instanciados con parámetros por defecto.")

# =====================================================================
# (7) ENTRENAMIENTO Y EVALUACIÓN CON CROSS-VALIDATION K=5
# =====================================================================
print("\n=== (7) VALIDACIÓN CRUZADA K=5 EN CONJUNTO DE ENTRENAMIENTO (80%) ===")

models = {
    'Ridge (Caja Blanca)': model_ridge,
    'Random Forest (Caja Negra 1)': model_rf,
    'XGBoost (Caja Negra 2)': model_xgb
}

cv_results = {}
scoring_metrics = {
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

for name, model in models.items():
    print(f"Ejecutando Cross-Validation para {name}...")
    scores = cross_validate(
        model, X_train, y_train, cv=5, scoring=scoring_metrics, n_jobs=-1
    )
    # Convertimos las métricas negativas a positivas para interpretación
    cv_results[name] = {
        'MAE_CV': -scores['test_mae'].mean(),
        'RMSE_CV': -scores['test_rmse'].mean(),
        'R2_CV': scores['test_r2'].mean()
    }

# =====================================================================
# (8) EVALUACIÓN EN EL CONJUNTO DE PRUEBA (TEST SET 20%)
# =====================================================================
print("\n=== (8) EVALUACIÓN EN EL CONJUNTO DE PRUEBA ===")

test_results = {}
predictions = {}

for name, model in models.items():
    print(f"Entrenando {name} sobre todo el conjunto de entrenamiento...")
    model.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Calcular métricas de regresión en test
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    test_results[name] = {
        'MAE_Test': mae,
        'RMSE_Test': rmse,
        'R2_Test': r2
    }

# =====================================================================
# (9) REPORTE COMPARATIVO DE MÉTRICAS (TABLA Y CONSOLA)
# =====================================================================
print("\n=== (9) REPORTE COMPARATIVO DE MÉTRICAS ===")

# Convertir diccionarios a DataFrames
df_cv = pd.DataFrame(cv_results).T
df_test = pd.DataFrame(test_results).T

# Unir ambos reportes
df_report = pd.concat([df_cv, df_test], axis=1)

print("\n--- TABLA COMPARATIVA DE MÉTRICAS DE REGRESIÓN ---")
print(df_report.round(4).to_string())

# =====================================================================
# (10) INTERFAZ/VISUALIZACIÓN DE RESULTADOS
# =====================================================================
print("\n=== (10) GENERANDO GRÁFICOS COMPARATIVOS ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

# 10.1 Gráfico 1: Real vs Predicho para Ridge (Caja Blanca)
ax_ridge = axes[0]
ax_ridge.scatter(y_test, predictions['Ridge (Caja Blanca)'], alpha=0.5, color='teal')
ax_ridge.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax_ridge.set_title('Ridge (Caja Blanca): Real vs Predicho')
ax_ridge.set_xlabel('Incidencia Real')
ax_ridge.set_ylabel('Incidencia Predicha')

# 10.2 Gráfico 2: Real vs Predicho para Random Forest (Caja Negra 1)
ax_rf = axes[1]
ax_rf.scatter(y_test, predictions['Random Forest (Caja Negra 1)'], alpha=0.5, color='navy')
ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax_rf.set_title('Random Forest (Caja Negra 1): Real vs Predicho')
ax_rf.set_xlabel('Incidencia Real')
ax_rf.set_ylabel('Incidencia Predicha')

# 10.3 Gráfico 3: Real vs Predicho para XGBoost (Caja Negra 2)
ax_xgb = axes[2]
ax_xgb.scatter(y_test, predictions['XGBoost (Caja Negra 2)'], alpha=0.5, color='darkorange')
ax_xgb.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax_xgb.set_title('XGBoost (Caja Negra 2): Real vs Predicho')
ax_xgb.set_xlabel('Incidencia Real')
ax_xgb.set_ylabel('Incidencia Predicha')

# 10.4 Gráfico 4: Comparación de R² en CV y Test
ax_metrics = axes[3]
metric_df = pd.DataFrame({
    'Modelos': ['Ridge', 'Random Forest', 'XGBoost'] * 2,
    'R2_Score': [cv_results['Ridge (Caja Blanca)']['R2_CV'],
                 cv_results['Random Forest (Caja Negra 1)']['R2_CV'],
                 cv_results['XGBoost (Caja Negra 2)']['R2_CV'],
                 test_results['Ridge (Caja Blanca)']['R2_Test'],
                 test_results['Random Forest (Caja Negra 1)']['R2_Test'],
                 test_results['XGBoost (Caja Negra 2)']['R2_Test']],
    'Tipo': ['CV (Train)'] * 3 + ['Test Set'] * 3
})

sns.barplot(data=metric_df, x='Modelos', y='R2_Score', hue='Tipo', palette='viridis', ax=ax_metrics)
ax_metrics.set_title('Comparación de R² (CV vs Test Set)')
ax_metrics.set_ylabel('Coeficiente de Determinación (R²)')
ax_metrics.set_ylim(-0.1, 1.1)

plt.tight_layout()
# Guardar gráfico en disco
PLOT_PATH = "dengue_regression_comparison.png"
plt.savefig(PLOT_PATH, dpi=150)
print(f"Gráficos comparativos guardados en: {os.path.abspath(PLOT_PATH)}")

print("\n=== PIPELINE DE REGRESIÓN EJECUTADO CON ÉXITO ===")
