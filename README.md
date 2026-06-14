# EpiPredict Dengue — Sistema Multi-Agente SMA-ML/DL

Sistema de alerta temprana para la predicción de dengue a escala subnacional en América Latina, desarrollado como Proyecto Final de Investigación. Combina Machine Learning (XGBoost) y Deep Learning (LSTM PyTorch) mediante una arquitectura de cinco agentes inteligentes coordinados.

---

## Descripción general

El sistema predice la tasa de incidencia de dengue (casos por 100,000 habitantes) a nivel departamental en América Latina utilizando datos epidemiológicos (OpenDengue) y climáticos (NASA POWER) del periodo 2014–2022. El backend expone una API REST desplegada en Railway; el frontend React está desplegado en Vercel. Los modelos y datos se almacenan en AWS S3.

---

## Arquitectura del sistema

```
Agente 1 (Ingesta)
  -> Agente 2 (Preprocesamiento + Feature Engineering)
    -> Agente 3 (XGBoost + SHAP)
    -> Agente 4 (LSTM PyTorch)
      -> Agente 5 (Orquestador: Ensemble + Alertas)
```

### Agente 1 — Ingesta de datos

Descarga y consolida datos epidemiológicos de OpenDengue y datos climáticos de NASA POWER (temperatura máxima/mínima, precipitación, humedad relativa). Produce CSVs estructurados en el prefijo `datos_crudos/` de S3.

### Agente 2 — Preprocesamiento y Feature Engineering

Limpia, normaliza y genera el dataset de features. Calcula tasas de incidencia por 100,000 habitantes, aplica lags temporales, rolling means, codificación cíclica de estacionalidad y vecinos espaciales. Produce dos salidas:

- `dataset_maestro_mensual_latam.csv` — 14 columnas base para inferencia y backend
- `dataset_features_latam.csv` — 34 features completas para entrenamiento de Agentes 3 y 4

**34 features generadas:**

| Grupo | Features |
|---|---|
| Base (6) | `agua_basica`, `tmax_promedio`, `tmin_promedio`, `precipitacion`, `humedad_promedio`, `densidad_poblacion` |
| Lags climaticos (12) | `tmax_lag1-3`, `tmin_lag1-3`, `precipitacion_lag1-3`, `humedad_lag1-3` |
| Lags de incidencia (6) | `incidencia_lag1-6` |
| Rolling means (2) | `incidencia_roll3`, `incidencia_roll6` |
| Vecinos espaciales (6) | `incidencia_vecinos_lag1-6` (promedio de 3 departamentos mas cercanos por GPS) |
| Estacionalidad ciclica (2) | `mes_sin`, `mes_cos` |

### Agente 3 — Predicción ML (XGBoost)

Entrena un modelo XGBoost con transformación logarítmica del target (`log1p`/`expm1`), genera importancias SHAP globales (TreeSHAP) y serializa todos los artefactos a S3.

- Hiperparámetros: `n_estimators=400`, `learning_rate=0.04`, `max_depth=6`, `random_state=42`
- Split: entrenamiento <= 2020, test 2021–2022
- **R² = 71.93% | MAE = 10.10 casos/100k**

### Agente 4 — Predicción DL (LSTM PyTorch)

Entrena una red LSTM apilada con PyTorch para capturar dependencias temporales de largo plazo.

- Arquitectura: `hidden_dim=64`, `num_layers=2`, `dropout=0.2`
- Lookback: 12 meses | Épocas: 80 | Optimizer: Adam (`lr=0.003`, `weight_decay=1e-4`)
- Features de entrada: `tmax_promedio`, `tmin_promedio`, `precipitacion`, `humedad_promedio`, `agua_basica`, `incidencia_dengue`
- `random_state=9`
- **R² = 76.50% | MAE = 10.22 casos/100k**

Genera `metrics.json` combinado con el R² de ensemble calculado honestamente (promediando predicciones en filas comunes del test set, no promediando R²s individuales).

### Agente 5 — Orquestador de Consenso (Ensemble + Alertas)

Combina las predicciones de Agentes 3 y 4 mediante promedio simple. Clasifica el nivel de riesgo epidemiológico con percentiles históricos calibrados por departamento.

- **R² ensemble = 75.62% | MAE = 9.86 casos/100k**
- Niveles de riesgo: Normal (< p25), Vigilancia (p25–p50), Alerta (p50–p90), Epidemia (> p90)
- En inferencia, los features `incidencia_vecinos_lag1-6` se computan usando los 3 departamentos geograficamente mas cercanos desde `departamentos_coordenadas.csv`, replicando exactamente el proceso de Agente 2.

---

## Métricas del sistema

| Modelo | R² | MAE (casos/100k) |
|---|---|---|
| XGBoost (Agente 3) | 71.93% | 10.10 |
| LSTM PyTorch (Agente 4) | 76.50% | 10.22 |
| Ensemble (Agente 5) | 75.62% | 9.86 |

Registros de entrenamiento: 12,180 observaciones mensuales (2014–2020). Test: 2021–2022.

---

## Stack tecnológico

| Capa | Tecnologías |
|---|---|
| Backend | FastAPI, Python 3.11, Uvicorn, Pydantic |
| ML / DL | XGBoost, PyTorch, Scikit-Learn, SHAP (TreeSHAP) |
| Datos | Pandas, NumPy, OpenDengue, NASA POWER API |
| Frontend | React 19, Vite, TailwindCSS, Leaflet.js |
| Infraestructura | Railway (backend), Vercel (frontend), AWS S3 (modelos y datos) |

---

## Estructura del repositorio

```
proyecto-final/
├── agentes/
│   ├── agente_1_recoleccion.py          # Ingesta OpenDengue + NASA POWER
│   ├── agente_2_preprocesamiento.py     # Feature engineering (34 features)
│   ├── agente_3_prediccion_ml.py        # Entrenamiento e inferencia XGBoost
│   ├── agente_4_prediccion_dl.py        # Entrenamiento e inferencia LSTM
│   ├── agente_5_alertas.py              # Orquestador ensemble + clasificacion riesgo
│   └── s3_client.py                     # Cliente S3 compartido
├── backend/
│   ├── main.py                          # FastAPI app + endpoints
│   ├── services.py                      # PredictionService (carga y orquesta agentes)
│   └── schemas.py                       # Modelos Pydantic request/response
├── frontend/
│   └── src/
│       ├── App.jsx                      # Raiz + PDF export
│       └── components/
│           ├── DashboardView.jsx        # Panel principal con mapa y estadisticas
│           ├── PredictorView.jsx        # Formulario de prediccion interactivo
│           ├── InfoView.jsx             # Documentacion tecnica del sistema
│           └── Topbar.jsx               # Barra de navegacion
├── Base de Datos/
│   ├── datos_crudos/                    # CSVs originales (gitignored)
│   ├── datos_procesados/                # Dataset maestro y de features (gitignored)
│   └── modelos/
│       ├── metrics.json                 # Metricas combinadas del sistema
│       └── ...                          # Artefactos de modelo (gitignored, en S3)
├── Dockerfile                           # python:3.11-slim, expone $PORT
├── requirements.txt
└── .env                                 # Credenciales AWS (no commiteado)
```

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS |
| `AWS_DEFAULT_REGION` | Region S3 (ej. `us-east-1`) |
| `S3_BUCKET` | Nombre del bucket (`epipredict-dengue`) |
| `PORT` | Puerto de la aplicacion (Railway lo inyecta automaticamente) |

---

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/health` | Estado del sistema |
| `GET` | `/api/metrics` | Metricas de los modelos (R², MAE) |
| `GET` | `/api/paises` | Lista de paises y departamentos disponibles |
| `POST` | `/api/predecir` | Prediccion de incidencia para un departamento |
| `GET` | `/api/historico/{iso_a0}/{adm_1_name}` | Serie historica mensual de un departamento |
| `GET` | `/api/top-departamentos` | Top 5 departamentos por incidencia media historica |
| `GET` | `/api/shap-global` | Importancias SHAP globales del modelo XGBoost |

---

## Fuentes de datos

- **OpenDengue Project** — Datos abiertos de vigilancia epidemiologica de dengue a escala subnacional en America Latina.
- **NASA POWER API** (NASA Langley Research Center) — Temperatura maxima/minima, precipitacion y humedad relativa historica mensual.
- **World Bank Open Data** — Estimaciones poblacionales anuales por pais y subregion.

---

## Notas de diseño

- El modelo LSTM usa exclusivamente 6 variables climaticas/epidemiologicas para las secuencias; el XGBoost usa las 34 features del dataset de features. Ambos comparten el mismo split temporal (train <= 2020, test 2021–2022).
- El ensemble R² (75.62%) se calcula promediando las predicciones de ambos modelos en el subconjunto de filas del test set que tienen correspondencia en ambos modelos, no promediando los R²s individuales.
- Durante inferencia, `incidencia_vecinos_lag1-6` se computa buscando los 3 departamentos geograficamente mas cercanos en `departamentos_coordenadas.csv` y promediando su incidencia historica real desde `df_master`, replicando el proceso de Agente 2 sin aproximaciones.
- El backend descarga todos los artefactos desde S3 al iniciar. Si S3 no esta disponible, usa los archivos locales en `Base de Datos/modelos/`.
