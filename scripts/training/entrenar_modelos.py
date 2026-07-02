# -*- coding: utf-8 -*-
"""
Script de entrenamiento completo — DenguePredict SMA-ML/DL
----------------------------------------------------------
Ejecuta el pipeline de entrenamiento con responsabilidades separadas:
  Agente 3 (solo ML: XGBoost + SHAP)  ─┐
                                        ├─► Agente 5 (semáforo/orquestador:
  Agente 4 (solo DL: LSTM)           ─┘    ensemble 0.5/0.5 + clasificación)

Los Agentes 3 y 4 entrenan de forma independiente entre sí — ninguno conoce
al otro. El Agente 5 es el único punto que combina ambos resultados y
escribe el metrics.json final. El Agente 6 ajusta los pesos base 0.5/0.5 de
forma dinámica en cada inferencia (no interviene en este script).

Hiperparametros de XGBoost y LSTM obtenidos mediante Bayesian Optimization
(Optuna TPESampler) con K=5 folds cronologicos, ver agente_3_prediccion_ml.py
y agente_4_prediccion_dl.py para el proceso completo.

Ejecutar: python entrenar_modelos.py
"""

import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.insert(0, os.path.join(BASE_DIR, "agents"))

from agente_3_prediccion_ml import AgentePrediccionML
from agente_4_prediccion_dl import AgentePrediccionDL
from agente_5_alertas import AgenteOrquestador

if __name__ == "__main__":
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETO — DenguePredict SMA-ML/DL")
    print("  Agente 3 (XGBoost) + Agente 4 (LSTM) -> Agente 5 (ensemble + alertas)")
    print("=" * 70)

    # Agente 3: XGBoost con hiperparametros optimizados por Bayesian Optimization (Optuna TPE)
    agente_ml = AgentePrediccionML(base_dir=BASE_DIR)
    metricas_ml = agente_ml.entrenar_modelo()

    # Agente 4: LSTM (Optuna TPE) — entrena de forma independiente del Agente 3
    agente_dl = AgentePrediccionDL(base_dir=BASE_DIR)
    metricas_dl = agente_dl.entrenar_modelo()

    # Agente 5: combina ambos modelos (ensemble 0.5/0.5 + clasificación 3 niveles)
    # y escribe el metrics.json final
    AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, BASE_DIR)

    print("\n  Entrenamiento completado.")
    print("  Artefactos guardados en data/models/ y subidos a S3.")
    print("=" * 70)
