# -*- coding: utf-8 -*-
"""
SMA-ML/DL - FastAPI Backend Entrypoint
--------------------------------------
Servidor REST asíncrono para interactuar con el Sistema Multi-Agente (SMA-ML/DL).
Carga el motor de agentes en memoria al arrancar y expone los servicios de predicción.
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

# Importar esquemas y servicios locales
from backend.schemas import (
    SimulationRequest, 
    RawPredictionRequest, 
    PredictionResponse, 
    HistoricalRecord,
    CountryMetadata
)
from backend.services import PredictionService

app = FastAPI(
    title="API del Sistema Multi-Agente SMA-ML/DL",
    description="Servicio Backend REST para predicción y análisis explicable de dengue en Latinoamérica.",
    version="1.0.0"
)

# Configurar políticas CORS para permitir llamadas desde el Frontend en Vercel o local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar a dominios específicos de Vercel en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicio de predicción en memoria (RAM)
# El servicio buscará de manera predeterminada el directorio base del proyecto
prediction_service = None

@app.on_event("startup")
def startup_event():
    global prediction_service
    # Detectar el directorio base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_service = PredictionService(base_dir=base_dir)

@app.get("/", tags=["General"])
def read_root():
    return {
        "status": "online",
        "proyecto": "Sistema Multi-Agente de Predicción de Dengue (SMA-ML/DL)",
        "autor": "Yeshua Chavez",
        "endpoints": [
            "GET /api/status",
            "GET /api/metadata",
            "GET /api/historical",
            "POST /api/predict/simulate",
            "POST /api/predict/raw",
            "GET /api/explain/global"
        ]
    }

@app.get("/api/status", tags=["General"])
def get_status():
    if prediction_service is None or prediction_service.df_master is None:
        return {"status": "loading", "message": "Inicializando modelos en memoria RAM..."}
    return {
        "status": "ready",
        "message": "Servidor listo y motor multi-agente en RAM.",
        "percentiles_incidencia": {
            "p25_bajo": prediction_service.p25,
            "p50_vigilancia": prediction_service.p50,
            "p90_alerta": prediction_service.p90
        }
    }

@app.get("/api/metadata", response_model=Dict[str, CountryMetadata], tags=["Datos"])
def get_metadata():
    try:
        metadata = prediction_service.obtener_metadatos_paises()
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener metadatos: {str(e)}")

@app.get("/api/coordinates", tags=["Datos"])
def get_coordinates():
    try:
        if prediction_service.df_coords is None:
            raise HTTPException(status_code=404, detail="Coordenadas no disponibles.")
        # Reemplazar NaNs para evitar errores de serialización JSON
        df_clean = prediction_service.df_coords.replace({np.nan: None})
        records = df_clean.to_dict(orient="records")
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener coordenadas: {str(e)}")

@app.get("/api/historical", response_model=List[HistoricalRecord], tags=["Datos"])
def get_historical(
    iso_a0: str = Query(..., description="Código ISO del país (ej. PER)"),
    adm_1_name: str = Query(..., description="Nombre del departamento/subregión")
):
    try:
        records = prediction_service.obtener_historico_departamento(iso_a0, adm_1_name)
        if not records:
            raise HTTPException(status_code=404, detail="No se encontraron registros históricos para los parámetros indicados.")
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener histórico: {str(e)}")

@app.post("/api/predict/simulate", response_model=PredictionResponse, tags=["Predicción"])
def predict_simulate(req: SimulationRequest):
    try:
        overrides = None
        if req.clima_overrides:
            overrides = {k: v for k, v in req.clima_overrides.items() if v is not None}
            
        res = prediction_service.simular_prediccion_departamento(
            iso_a0=req.iso_a0,
            adm_1_name=req.adm_1_name,
            ano=req.ano,
            mes=req.mes,
            clima_overrides=overrides,
            compute_shap=req.include_shap,
        )
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en simulación de predicción: {str(e)}")

@app.post("/api/predict/raw", response_model=PredictionResponse, tags=["Predicción"])
def predict_raw(req: RawPredictionRequest):
    try:
        if len(req.features) != len(prediction_service.cols_feat):
            raise HTTPException(
                status_code=400, 
                detail=f"Dimensiones de entrada inválidas. Se requieren exactamente {len(prediction_service.cols_feat)} variables predictoras."
            )
        res = prediction_service.realizar_prediccion_vector(req.features)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar predicción cruda: {str(e)}")

@app.get("/api/metrics", tags=["General"])
def get_metrics():
    import json as _json
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Base de Datos", "modelos", "metrics.json"
    )
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Archivo de métricas no encontrado.")
    with open(metrics_path, "r") as f:
        return _json.load(f)

@app.get("/api/top-departments", tags=["Datos"])
def get_top_departments(n: int = Query(5, ge=1, le=20, description="Número de departamentos a retornar")):
    try:
        return prediction_service.obtener_top_departamentos(n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener top departamentos: {str(e)}")

@app.get("/api/features", tags=["Datos"])
def get_features(
    iso_a0: str = Query(..., description="Código ISO del país (ej. PER)"),
    adm_1_name: str = Query(..., description="Nombre del departamento/subregión"),
    ano: int = Query(None, description="Año (opcional, usa el último disponible si se omite)"),
    mes: int = Query(None, description="Mes (opcional, usa el último disponible si se omite)"),
):
    """Devuelve el vector de 34 features del último período disponible sin correr ningún modelo."""
    try:
        return prediction_service.obtener_features_departamento(iso_a0, adm_1_name, ano, mes)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener features: {str(e)}")

@app.get("/api/explain/global", tags=["Explicabilidad XAI"])
def explain_global():
    if prediction_service.shap_importance is None:
        raise HTTPException(status_code=404, detail="Explicabilidad SHAP global no disponible.")
    return prediction_service.shap_importance

# Iniciar servidor local si se ejecuta de forma directa
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
