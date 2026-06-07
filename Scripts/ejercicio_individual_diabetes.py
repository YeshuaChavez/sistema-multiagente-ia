# -*- coding: utf-8 -*-
"""
Sistema de Machine Learning para la Clasificación Binaria de Diabetes
Dataset: Pima Indians Diabetes (Kaggle)
Institución: Universidad Nacional Mayor de San Marcos (UNMSM)
Curso: Aprendizaje Supervisado / Inteligencia Artificial

Modelos implementados:
- Caja Blanca: Regresión Logística y Naive Bayes (Gaussian)
- Caja Negra: Support Vector Machine (SVM) y Red Neuronal Artificial (MLP)
"""

# =====================================================================
# (1) IMPORTACIONES
# =====================================================================
import urllib.request
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Configuración de estética de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# =====================================================================
# (2) CARGA Y EXPLORACIÓN DEL DATASET
# =====================================================================
print("=== (2) CARGA Y EXPLORACIÓN DEL DATASET ===")
# URL del dataset Pima Indians Diabetes
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATASET_PATH = "pima-indians-diabetes.csv"

# Descargar el dataset si no existe localmente
if not os.path.exists(DATASET_PATH):
    print(f"Descargando dataset desde: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print("Descarga completa.")

# Definir los nombres de las columnas
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Leer el archivo CSV
df = pd.read_csv(DATASET_PATH, names=columns)
print(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas.")
print("\nPrimeras 5 filas:")
print(df.head())

print("\nDistribución de la variable objetivo (Outcome):")
print(df['Outcome'].value_counts(normalize=True))

# =====================================================================
# (3) PREPROCESAMIENTO DE VARIABLES
# =====================================================================
print("\n=== (3) PREPROCESAMIENTO DE VARIABLES ===")
# Nota: En este dataset, valores de 0 en Glucose, BloodPressure, SkinThickness,
# Insulin y BMI son biológicamente imposibles y representan valores faltantes (NaN).
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"Valores en cero antes del reemplazo:")
for col in cols_with_zeros:
    print(f"  - {col}: {(df[col] == 0).sum()} valores en cero.")

# Reemplazar ceros por NaN en las columnas indicadas
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Separar características (X) y etiqueta (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Imputación de valores faltantes usando la mediana
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Escalado/Estandarización de variables (muy importante para KNN, SVM, Reg. Logística y Redes Neuronales)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print("\nPreprocesamiento finalizado:")
print("  - Ceros biológicamente inconsistentes reemplazados por la mediana de la columna.")
print("  - Variables estandarizadas a media=0 y desviación estándar=1.")

# =====================================================================
# (4) PARTICIÓN EN ENTRENAMIENTO (80%) Y PRUEBA (20%)
# =====================================================================
print("\n=== (4) PARTICIÓN 80/20 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} registros.")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} registros.")

# =====================================================================
# (5) MODELOS CAJA BLANCA CON GRID SEARCH
# =====================================================================
print("\n=== (5) ENTRENAMIENTO MODELOS CAJA BLANCA (GRID SEARCH) ===")

# --- Modelo 1: Regresión Logística ---
print("Entrenando Regresión Logística...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga']
}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"  - Mejores hiperparámetros LR: {grid_lr.best_params_}")

# --- Modelo 2: Gaussian Naive Bayes ---
print("Entrenando Gaussian Naive Bayes...")
gnb = GaussianNB()
# Naive Bayes no tiene muchos hiperparámetros, optimizamos 'var_smoothing'
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=20)
}
grid_nb = GridSearchCV(gnb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)
grid_nb.fit(X_train, y_train)
best_nb = grid_nb.best_estimator_
print(f"  - Mejores hiperparámetros NB: {grid_nb.best_params_}")

# =====================================================================
# (6) MODELOS CAJA NEGRA CON GRID SEARCH
# =====================================================================
print("\n=== (6) ENTRENAMIENTO MODELOS CAJA NEGRA (GRID SEARCH) ===")

# --- Modelo 3: Support Vector Machine (SVM) ---
print("Entrenando Support Vector Machine (SVM)...")
svm = SVC(probability=True, random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print(f"  - Mejores hiperparámetros SVM: {grid_svm.best_params_}")

# --- Modelo 4: Red Neuronal Artificial (ANN / MLP) ---
print("Entrenando Red Neuronal (MLP Classifier)...")
mlp = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
param_grid_mlp = {
    'hidden_layer_sizes': [(32,), (64, 32), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}
grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(X_train, y_train)
best_mlp = grid_mlp.best_estimator_
print(f"  - Mejores hiperparámetros MLP: {grid_mlp.best_params_}")

# =====================================================================
# (7) CROSS-VALIDATION K=5 PARA EVALUACIÓN MÚLTIPLE
# =====================================================================
print("\n=== (7) VALIDACIÓN CRUZADA K=5 PARA LOS MEJORES ESTIMADORES ===")

models = {
    'Regresión Logística': best_lr,
    'Naive Bayes': best_nb,
    'SVM': best_svm,
    'Red Neuronal (MLP)': best_mlp
}

cv_results = {}
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

for name, model in models.items():
    print(f"Ejecutando Cross-Validation para {name}...")
    scores = cross_validate(
        model, X_train, y_train, cv=5, scoring=scoring_metrics, n_jobs=-1
    )
    cv_results[name] = {
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'F1': scores['test_f1'].mean()
    }

# =====================================================================
# (8) EVALUACIÓN EN EL CONJUNTO DE PRUEBA (TEST SET)
# =====================================================================
print("\n=== (8) EVALUACIÓN EN EL CONJUNTO DE PRUEBA ===")

test_results = {}

for name, model in models.items():
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    test_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    }

# =====================================================================
# (9) REPORTE COMPARATIVO DE MÉTRICAS
# =====================================================================
print("\n=== (9) REPORTE COMPARATIVO DE MÉTRICAS ===")

# Crear DataFrames para mostrar resultados
df_cv = pd.DataFrame(cv_results).T
df_test = pd.DataFrame(test_results).T

print("\n--- MÉTRICAS PROMEDIO DE VALIDACIÓN CRUZADA (K=5) ON TRAIN (80%) ---")
print(df_cv.round(4).to_string())

print("\n--- MÉTRICAS EN EL CONJUNTO DE PRUEBA (TEST SET 20%) ---")
print(df_test.round(4).to_string())

# =====================================================================
# (10) INTERFAZ/VISUALIZACIÓN DE RESULTADOS
# =====================================================================
print("\n=== (10) GENERANDO GRÁFICOS COMPARATIVOS ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

# 10.1 Curvas ROC (Receiver Operating Characteristic)
ax_roc = axes[0]
ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Clasificador Aleatorio')

for name, model in models.items():
    # Obtener probabilidades de predicción para la clase 1
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else: # Fallback para modelos sin predict_proba directa
        y_prob = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax_roc.set_title('Curvas ROC (Test Set)')
ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax_roc.legend(loc='lower right')

# 10.2 Matriz de Confusión para cada modelo (los otros 3 subplots)
confusion_models = [('Regresión Logística', best_lr), ('SVM', best_svm), ('Red Neuronal (MLP)', best_mlp)]

for i, (name, model) in enumerate(confusion_models, start=1):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    ax_cm = axes[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
    ax_cm.set_title(f'Matriz de Confusión: {name}')
    ax_cm.set_xlabel('Predicción')
    ax_cm.set_ylabel('Realidad')
    ax_cm.set_xticklabels(['No Diabético', 'Diabético'])
    ax_cm.set_yticklabels(['No Diabético', 'Diabético'], rotation=0)

plt.tight_layout()
# Guardar gráfico en disco
PLOT_PATH = "diabetes_model_comparison.png"
plt.savefig(PLOT_PATH, dpi=150)
print(f"Gráfico guardado exitosamente como: {os.path.abspath(PLOT_PATH)}")

print("\n=== PIPELINE EJECUTADO CON ÉXITO ===")
