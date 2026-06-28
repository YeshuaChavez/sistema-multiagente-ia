# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 6: Detección de Régimen Epidémico
--------------------------------------------------
Responsabilidad: Clasificar el estado epidémico actual del departamento
en uno de cinco regímenes (Normal, Vigilancia, Pre-brote, Brote activo,
Post-pico) y ajustar los pesos del ensemble del Agente 5 en consecuencia.

Fundamento: XGBoost (basado en árboles) no puede extrapolar más allá del
rango visto en entrenamiento, subestimando brotes extremos. LSTM captura
mejor el momentum temporal en escenarios de alta incidencia sostenida.
Este agente detecta el régimen y balancea ambos modelos dinámicamente.
"""

import numpy as np


class AgenteRegimen:
    """
    Agente 6 — Detección de Régimen Epidémico.

    Clasifica el estado epidémico usando percentiles locales e indicadores
    de tendencia, y retorna pesos ajustados para el ensemble (Agente 5).

    Regímenes:
      Normal      — incidencia dentro del rango histórico habitual
      Vigilancia  — incidencia moderada o descendiendo desde nivel alto
      Pre-brote   — incidencia en ascenso antes de superar p90
      Brote activo — incidencia > p90 con tendencia positiva (LSTM domina)
      Post-pico   — incidencia > p90 pero descendiendo (XGBoost domina)
    """

    REGIMENES = {
        "Normal":        "Incidencia dentro del rango histórico normal.",
        "Vigilancia":    "Incidencia moderada o en descenso. Monitoreo preventivo.",
        "Pre-brote":     "Tendencia ascendente detectada. Alerta preventiva recomendada.",
        "Brote activo":  "Brote confirmado en fase activa. Respuesta inmediata.",
        "Post-pico":     "Brote en declive. Continuar vigilancia hasta normalización.",
    }

    def __init__(self, w_xgb_base: float, w_lstm_base: float):
        self.w_xgb_base  = w_xgb_base
        self.w_lstm_base = w_lstm_base

    def detectar(
        self,
        lag1_raw:   float,
        lag1_log:   float,
        lag2_log:   float,
        p25_local:  float,
        p50_local:  float,
        p90_local:  float,
    ) -> dict:
        """
        Detecta el régimen epidémico y calcula pesos ajustados del ensemble.

        Args:
            lag1_raw:  Incidencia real del mes anterior (casos/100k, escala original)
            lag1_log:  log1p(lag1_raw) — valor en escala de entrenamiento
            lag2_log:  log1p(lag2_raw) — mes t-2 en escala de entrenamiento
            p25/50/90_local: Percentiles históricos del departamento

        Returns:
            dict con: regimen, descripcion, w_xgb, w_lstm
        """
        tendencia = lag1_log - lag2_log   # >0 sube, <0 baja
        p90_ref   = max(p90_local, 1.0)

        if lag1_raw <= max(p25_local, 0.0):
            regimen = "Normal"
            w_xgb, w_lstm = self.w_xgb_base, self.w_lstm_base

        elif lag1_raw <= p50_local:
            regimen = "Vigilancia"
            w_xgb, w_lstm = self.w_xgb_base, self.w_lstm_base

        elif lag1_raw <= p90_local:
            if tendencia > 0:
                regimen = "Pre-brote"
                w_lstm = min(self.w_lstm_base * 1.4, 0.65)
                w_xgb  = 1.0 - w_lstm
            else:
                regimen = "Vigilancia"
                w_xgb, w_lstm = self.w_xgb_base, self.w_lstm_base

        else:
            if tendencia > 0:
                regimen     = "Brote activo"
                extremeness = min(lag1_raw / p90_ref, 3.0)
                w_lstm      = min(self.w_lstm_base * extremeness, 0.80)
                w_xgb       = 1.0 - w_lstm
            else:
                regimen = "Post-pico"
                w_xgb   = min(self.w_xgb_base * 1.5, 0.75)
                w_lstm  = 1.0 - w_xgb

        return {
            "regimen":     regimen,
            "descripcion": self.REGIMENES[regimen],
            "w_xgb":       round(w_xgb, 4),
            "w_lstm":       round(w_lstm, 4),
        }
