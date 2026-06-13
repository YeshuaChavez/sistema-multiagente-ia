# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Backend Schemas
---------------------------
Modelos de validación Pydantic para peticiones y respuestas HTTP.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ClimaOverrides(BaseModel):
    tmax_promedio: Optional[float] = Field(None, description="Temperatura Máxima promedio en °C")
    tmin_promedio: Optional[float] = Field(None, description="Temperatura Mínima promedio en °C")
    precipitacion: Optional[float] = Field(None, description="Precipitación acumulada en mm")
    humedad_promedio: Optional[float] = Field(None, description="Humedad Relativa promedio en %")

class SimulationRequest(BaseModel):
    iso_a0: str = Field(..., example="PER", description="Código ISO del país (3 letras)")
    adm_1_name: str = Field(..., example="LORETO", description="Nombre del departamento/subregión")
    ano: Optional[int] = Field(None, example=2022, description="Año de referencia (opcional; si se omite usa el último registro disponible)")
    mes: Optional[int] = Field(None, example=6, description="Mes de referencia (1-12, opcional)")
    clima_overrides: Optional[Dict[str, float]] = Field(None, description="Valores simulados por el usuario para cualquier variable")

class RawPredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Vector de 23 características ordenadas")

class RiskLevel(BaseModel):
    nivel: str = Field(..., description="Descripción del riesgo: Normal, Vigilancia, Alerta, Epidemia")
    codigo: str = Field(..., description="Etiqueta corta del riesgo")
    color: str = Field(..., description="Color hexadecimal para renderizado en frontend")

class PredictionResponse(BaseModel):
    prediccion_ml: float = Field(..., description="Predicción del Agente 3 (XGBoost)")
    riesgo_ml: RiskLevel = Field(..., description="Nivel de riesgo estimado por XGBoost")
    prediccion_dl: float = Field(..., description="Predicción del Agente 4 (PyTorch MLP)")
    riesgo_dl: RiskLevel = Field(..., description="Nivel de riesgo estimado por la MLP")
    prediccion_ensemble: float = Field(..., description="Predicción promediada del Agente 5 (Ensemble)")
    riesgo_ensemble: RiskLevel = Field(..., description="Nivel de riesgo de la predicción final")
    features_usadas: Optional[Dict[str, float]] = Field(None, description="Características reales o medianas utilizadas como entrada")
    percentiles_locales: Optional[Dict[str, float]] = Field(None, description="Percentiles locales de incidencia (p25, p50, p90)")

class HistoricalRecord(BaseModel):
    fecha: str
    ano: int
    mes: int
    casos: int
    incidencia: float
    tmax: float
    tmin: float
    precipitacion: float
    humedad: float

class CountryMetadata(BaseModel):
    iso_a0: str
    departamentos: List[str]
