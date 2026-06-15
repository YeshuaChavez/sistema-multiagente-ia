# -*- coding: utf-8 -*-
"""
Script de entrenamiento completo — DenguePredict SMA-ML/DL
----------------------------------------------------------
Ejecuta el pipeline de entrenamiento secuencial:
  Agente 3 (XGBoost + SHAP) -> Agente 4 (LSTM + pesos ensemble)

Hiperparametros de XGBoost obtenidos mediante GridSearchCV + TimeSeriesSplit
(ver optimizar_hiperparametros.py para el proceso completo).

Ejecutar: python entrenar_modelos.py
"""

import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.insert(0, os.path.join(BASE_DIR, "agentes"))

from agente_3_prediccion_ml import AgentePrediccionML
from agente_4_prediccion_dl import AgentePrediccionDL

if __name__ == "__main__":
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETO — DenguePredict SMA-ML/DL")
    print("  Agente 3 (XGBoost) -> Agente 4 (LSTM) -> Ensemble")
    print("=" * 70)

    # Agente 3: XGBoost con hiperparametros optimizados por GridSearchCV
    agente_ml = AgentePrediccionML(base_dir=BASE_DIR)
    metricas_ml = agente_ml.entrenar_modelo()

    # Agente 4: LSTM + calcula pesos optimos del ensemble
    agente_dl = AgentePrediccionDL(base_dir=BASE_DIR)
    agente_dl.entrenar_modelo(metricas_ml=metricas_ml)

    print("\n  Entrenamiento completado.")
    print("  Artefactos guardados en Base de Datos/modelos/ y subidos a S3.")
    print("=" * 70)
