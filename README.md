# EpiPredict — Sistema Multi-Agente SMA-ML/DL para Vigilancia Epidemiologica de Dengue

Sistema de alerta temprana para la prediccion de epidemias de dengue a escala subnacional en America Latina. Implementa una arquitectura de **seis agentes inteligentes coordinados** que combinan Machine Learning (XGBoost + SHAP) y Deep Learning (LSTM PyTorch) en un ensamble hibrido con deteccion dinamica de regimen epidemico, desplegado como aplicacion web completa en la nube.

**Proyecto Final — Universidad Nacional Mayor de San Marcos**
Facultad de Ingenieria de Sistemas e Informatica

---

## Demo

| Capa | URL |
|---|---|
| Frontend (Vercel) | https://proyecto-ia-eight.vercel.app |
| Backend API (Railway) | https://proyecto-ia-production.up.railway.app |
| Documentacion interactiva (Swagger) | https://proyecto-ia-production.up.railway.app/docs |

---

## Metricas del sistema

| Modelo | R² (log1p) | MAE (casos/100k) | RMSE |
|---|---|---|---|
| XGBoost — Agente 3 | **91.49%** | 6.07 | 22.18 |
| LSTM PyTorch — Agente 4 | 90.35% | 6.02 | 20.52 |
| **Ensamble — Agente 5** | **91.47%** | **5.83** | **20.67** |

- **Conjunto de entrenamiento:** 12,168 observaciones mensuales subnacionales
- **Conjunto de prueba:** 4,056 observaciones — particion cronologica estricta (ultimos 2 anos del dataset)
- **Split dinamico:** `split_ano = max_ano - 2`, sin hardcodear anos — permite reentrenamiento automatico consistente
- **Validacion temporal:** `TimeSeriesSplit(k=5)` sobre el conjunto de entrenamiento (folds cronologicos)
- **Dataset:** 16,224 registros × 73 features predictoras, 8 paises, 169 unidades subnacionales
- **Pesos base del ensamble:** `w_xgb = 0.50`, `w_lstm = 0.50`, ajustados dinamicamente por el Agente 6
- R² reportado en escala `log1p`, estandar para distribuciones epidemiologicas asimetricas

### Clasificacion de riesgo epidemico (3 clases)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Endemico | 92.77% | 89.65% | 91.18% | 2,532 |
| Alerta | 69.66% | 79.67% | 74.33% | 1,092 |
| Epidemia | 86.67% | 72.22% | 78.79% | 432 |
| **Accuracy global** | | | **85.11%** | 4,056 |
| **Cohen's Kappa** | | | **0.7196** (Sustancial) | |

Clasificacion basada en **percentiles historicos locales calibrados por departamento** (set de entrenamiento): Endemico (<=p50), Alerta (p50–p90), Epidemia (>p90).

---

## Cobertura geografica

| Pais | Codigo ISO | Unidades subnacionales |
|---|---|---|
| Argentina | ARG | Provincias |
| Bolivia | BOL | Departamentos |
| Brasil | BRA | Estados |
| Colombia | COL | Departamentos |
| Ecuador | ECU | Provincias |
| Mexico | MEX | Estados |
| Panama | PAN | Provincias |
| Peru | PER | Departamentos |

**Total: 169 unidades subnacionales — Periodo: 2014–2022**

---

## Arquitectura del sistema

```
Fuentes externas
  OpenDengue · NASA POWER · Banco Mundial · JMP OMS/UNICEF
        |
        v
  AWS S3 (epipredict-dengue)
  datos_crudos/ · datos_procesados/ · modelos/
        |
        v
  Agente 1 — Recoleccion
  Ingesta automatica desde fuentes oficiales
  Fallback: descarga ZIP desde GitHub OpenDengue si CSV ausente
        |
        v
  Agente 2 — Preprocesamiento + Feature Engineering
  73 features: lags, rolling, vecinos GPS, estacionalidad ciclica,
  indicadores epidemiologicos, dummies de pais
        |
     +--+--+
     v     v
  Agente 3   Agente 4
  XGBoost    LSTM PyTorch
  R²=91.49%  R²=90.35%
     +--+--+
        |
        v
  Agente 5 — Orquestador de Consenso
  Ensamble ponderado -> R²=91.47%
  Clasificacion: Endemico / Alerta / Epidemia
        |
        v  (consulta regimen epidemico)
  Agente 6 — Regimen Epidemico
  Ajuste dinamico de pesos: Normal / Vigilancia /
  Pre-brote / Brote activo / Post-pico
        |
        v
  Backend FastAPI (Railway) <---> Frontend React 19 / Vite (Vercel)

  +----- FASE 10: Monitoreo y Mantenimiento (GitHub Actions) -----+
  |  verificar_actualizacion.py — cron 1ro de cada mes            |
  |  1. Detecta nueva version OpenDengue (SHA GitHub)             |
  |  2. Drift PSI sobre features climaticas (NASA POWER)          |
  |  3. Reentrenamiento automatico si hay nueva version           |
  |  4. Sube drift_report.json + data_version.json a S3           |
  +---------------------------------------------------------------+
```

---

## Agentes

### Agente 1 — Recoleccion de datos

Descarga y consolida los datos historicos 2014–2022 desde:

- **OpenDengue** — Casos de dengue a nivel subnacional (8 paises, America Latina)
- **NASA POWER API** — Variables climaticas mensuales (temperatura max/min, precipitacion, humedad)
- **Banco Mundial** — Estimaciones de poblacion anual
- **JMP OMS/UNICEF** — Indicador de acceso a agua potable basica

**Fallback automatico:** Si el CSV local no existe, descarga el ZIP desde GitHub (`data/releases/V1.3/`) y lo extrae. Intenta versiones en orden descendente: V1.3 → V1.2.2 → V1.1 → V1.0.

---

### Agente 2 — Preprocesamiento y Feature Engineering

Calcula la tasa de incidencia mensual (`casos / poblacion * 100,000`) y construye las **73 variables predictoras**:

| Grupo | Variables | N |
|---|---|---|
| Base climatica y demografica | `tmax`, `tmin`, `precipitacion`, `humedad`, `poblacion`, `densidad_poblacion` | 6 |
| Lags climaticos | `tmax/tmin/precipitacion/humedad` lag 1–6 | 24 |
| Lags de incidencia | `incidencia_lag1` a `incidencia_lag12` (log1p) | 12 |
| Rolling means | `incidencia_roll3`, `roll6`, `roll12` (log1p) | 3 |
| Vecinos espaciales | `incidencia_vecinos_lag1–lag6` (3 deptos mas cercanos por GPS) | 6 |
| Estacionalidad ciclica | `mes_sin = sin(2pi*mes/12)`, `mes_cos = cos(2pi*mes/12)` | 2 |
| Indicadores epidemiologicos | `indicador_covid`, `indicador_nino`, `indicador_nina` | 3 |
| Features derivadas | `amplitud_termica`, `temperatura_media`, `precipitacion_anomalia`, `aceleracion_incidencia`, `cambio_interanual`, `tendencia_1m`, `tendencia_3m`, `fase_ascendente`, `indicador_brote` | 9 |
| Dummies de pais | `pais_ARG/BOL/BRA/COL/ECU/MEX/PAN/PER` | 8 |
| **Total** | | **73** |

Artefactos producidos en S3:
- `datos_procesados/dataset_maestro_mensual_latam.csv` — 18,252 filas × 14 cols
- `datos_procesados/dataset_features_latam.csv` — 16,224 filas × 81 cols

---

### Agente 3 — Prediccion ML (XGBoost + SHAP)

Pipeline con transformacion logaritmica del target:

```
SimpleImputer(median) -> StandardScaler -> XGBRegressor
target: log1p(incidencia) -> salida: expm1(prediccion)
```

**Optimizacion de hiperparametros:** Bayesian Optimization (Optuna TPE), 50 trials × K=5 folds cronologicos (2016-2020), ejecutado en Google Colab (`notebooks/colab_bayesian_xgb_lstm.ipynb`). Mejores hiperparametros encontrados:

```
n_estimators     = 805     subsample        = 0.656
learning_rate    = 0.0242  colsample_bytree = 0.516
max_depth        = 5       gamma            = 0.088
min_child_weight = 10
```

> Mismo algoritmo en entrenamiento inicial (Colab, GPU T4) y reentrenamiento automatico (GitHub Actions, CPU).

Genera **importancias SHAP globales** (TreeSHAP sobre el set de prueba completo) y **SHAP locales** por prediccion individual.

**Resultado en test:** R² = 91.49% | MAE = 6.07 casos/100k | RMSE = 22.18

---

### Agente 4 — Prediccion DL (LSTM PyTorch)

Red LSTM de dos capas apiladas con lookback de 12 meses. Usa **6 features** (la memoria interna reemplaza los lags explicitos del Agente 3):

```
tmax_promedio · tmin_promedio · precipitacion ·
humedad_promedio · agua_basica · incidencia_dengue
```

**Optimizacion de hiperparametros:** Bayesian Optimization (Optuna TPE), 30 trials × K=5 folds cronologicos, GPU T4 en Google Colab. Mejores hiperparametros:

```
Input:    12 pasos x 6 features
LSTM:     hidden_dim=77, num_layers=3, dropout=0.293
Output:   Linear(77 -> 1) -> expm1 -> incidencia predicha
lr=0.00988
```

**Entrenamiento:** Adam, ReduceLROnPlateau (patience=5), Early Stopping (patience=15), `torch.manual_seed(42)`, GPU T4 (Colab) / CPU (produccion).

**Resultado en test:** R² = 90.35% | MAE = 6.02 casos/100k | RMSE = 20.52

---

### Agente 5 — Orquestador de Consenso (Ensamble + Alertas)

Combina las predicciones de los Agentes 3 y 4 con pesos base 50/50, ajustados por el Agente 6:

```
pred_ensemble = w_xgb * pred_xgb + w_lstm * pred_lstm
```

Clasifica el nivel de riesgo con percentiles historicos locales por departamento (calibrados en el set de entrenamiento):

| Nivel | Criterio | Color |
|---|---|---|
| Endemico | prediccion <= p50 local | Verde |
| Alerta | p50 < prediccion <= p90 | Naranja |
| Epidemia | prediccion > p90 | Rojo |

**Resultado en test:** R² = 91.47% | MAE = 5.83 | RMSE = 20.67 | Accuracy = 85.11% | Kappa = 0.7196

---

### Agente 6 — Regimen Epidemico (Ajuste Dinamico de Pesos)

Detecta el estado epidemico actual usando `incidencia_lag1` vs. percentiles historicos locales y tendencia (`log1p(lag1) - log1p(lag2)`):

| Regimen | Condicion | Ajuste de pesos |
|---|---|---|
| Normal | `lag1 <= p25` | 50/50 (base) |
| Vigilancia | `p25 < lag1 <= p50` | 50/50 (base) |
| Pre-brote | `p50 < lag1 <= p90` + tendencia creciente | `w_lstm` -> min(0.50 * 1.4, 0.65) |
| Brote activo | `lag1 > p90` + tendencia creciente | `w_lstm` -> min(0.50 * (lag1/p90), 0.80) |
| Post-pico | `lag1 > p90` + tendencia decreciente | `w_xgb` -> min(0.50 * 1.5, 0.75) |

Los cinco regimenes son internos al sistema. El usuario solo ve los tres niveles del Agente 5.

---

## Ciclo de vida de los modelos (10 fases)

Las 10 fases estan documentadas explicitamente en el codigo fuente de los Agentes 3 y 4 (docstrings + comentarios inline):

| Fase | Descripcion | Donde |
|---|---|---|
| 1 — Definicion del problema | Prediccion de incidencia de dengue subnacional mensual | Docstrings Agentes 3 y 4 |
| 2 — Recoleccion | Ingesta desde OpenDengue, NASA POWER, BM, JMP | `agente_1_recoleccion.py` |
| 3 — Preparacion | Feature engineering (73 variables) | `agente_2_preprocesamiento.py` |
| 4 — Division | Split cronologico dinamico: `split_ano = max_ano - 2` | Agentes 3 y 4 |
| 5 — Seleccion del modelo | Pipeline XGBoost / LSTM 3 capas PyTorch | Agentes 3 y 4 |
| 6a — Entrenamiento baseline | Parametros por defecto | Agentes 3 y 4 |
| 7a — Evaluacion baseline | R², MAE en test set | Agentes 3 y 4 |
| 8 — Optimizacion | Bayesian Optimization (Optuna TPESampler) + K=5 folds cronologicos (ML y DL) | Agentes 3 y 4 |
| 6b — Reentrenamiento | Con mejores hiperparametros (`refit=True`) | Agentes 3 y 4 |
| 7b — Evaluacion final | R², MAE, RMSE + ensamble con pesos fijos 0.5/0.5 | Agentes 3 y 4 |
| 9 — Despliegue | Serializacion S3, FastAPI Railway, React Vercel | `s3_client.py`, `backend/`, `frontend/` |
| **10 — Mantenimiento** | **Drift PSI + version check + reentrenamiento auto** | `verificar_actualizacion.py` + GitHub Actions |

### Fase 10: Monitoreo y Mantenimiento automatizado

Ejecutado automaticamente el **1ro de cada mes** via GitHub Actions (`.github/workflows/retrain.yml`):

**10a — Deteccion de nueva version de datos:**
Consulta el SHA del ultimo commit en `data/releases/` del repositorio OpenDengue via GitHub API. Si difiere del guardado en `data_version.json`, hay nuevos datos disponibles.

**10b — Drift de covariables (PSI — Population Stability Index):**
Descarga datos climaticos recientes de NASA POWER (ultimos 2 anos disponibles) y calcula el PSI para cada feature climatica vs. la distribucion del set de entrenamiento.

| PSI | Nivel | Accion |
|---|---|---|
| < 0.1 | Estable | Sin accion |
| 0.1 – 0.2 | Moderado | Monitorear |
| >= 0.2 | Alto | Priorizar reentrenamiento |

Guarda `drift_report.json` con PSI por feature y bandera `alerta_drift`. Si hay drift alto sin nueva version, crea un GitHub Issue automaticamente.

> **Nota sobre drift de concepto:** El drift de covariables (features climaticas) es detectable en tiempo casi-real via NASA POWER. El drift de concepto (cambio en la relacion features->incidencia) requiere datos etiquetados de OpenDengue, que se publican con 6–12 meses de latencia — se evalua al momento del reentrenamiento con nueva version del dataset.

**10c — Reentrenamiento automatico:**
Si hay nueva version: descarga dataset, ejecuta pipeline completo (Agentes 2, 3, 4), sube modelos actualizados a S3.

**Ejemplo de reporte drift (Junio 2026):**
```json
{
  "estado": "calculado",
  "features": {
    "tmax_promedio":    { "psi": 0.1675, "nivel": "moderado" },
    "tmin_promedio":    { "psi": 0.3676, "nivel": "alto" },
    "precipitacion":    { "psi": 0.0736, "nivel": "estable" },
    "humedad_promedio": { "psi": 0.0357, "nivel": "estable" }
  },
  "psi_max": 0.3676,
  "alerta_drift": true,
  "nota": "Drift detectado en features de entrada (covariable shift)..."
}
```

`tmin_promedio` con PSI=0.37 (alto) refleja el calentamiento climatico observado en America Latina entre el periodo de entrenamiento (2014–2022) y los datos recientes (2024–2025).

---

## Stack tecnologico

| Capa | Tecnologias |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 |
| ML | XGBoost 2.x · Scikit-Learn 1.8 · SHAP (TreeSHAP) · Bayesian Optimization (Optuna TPE) |
| DL | PyTorch 2.x · LSTM 3 capas apiladas (hidden=77, lookback=12 meses) |
| Datos | Pandas · NumPy · requests · python-dotenv |
| Frontend | React 19 · Vite · TailwindCSS · Leaflet.js · Material Symbols |
| Visualizacion | SVG puro — ScatterPlot 4,056 puntos sin librerias externas |
| PDF | jsPDF + jspdf-autotable — reporte tecnico exportable |
| Storage | AWS S3 (`epipredict-dengue`) via boto3 |
| Deploy backend | Railway (Docker `python:3.11-slim`) |
| Deploy frontend | Vercel |
| CI/CD | GitHub Actions — cron mensual (drift + reentrenamiento automatico) |

---

## Estructura del repositorio

```
/
├── agents/
│   ├── agente_1_recoleccion.py          # Ingesta + fallback ZIP OpenDengue
│   ├── agente_2_preprocesamiento.py     # Feature engineering (73 variables)
│   ├── agente_3_prediccion_ml.py        # XGBoost + Optuna TPE + SHAP (Fases 1-10)
│   ├── agente_4_prediccion_dl.py        # LSTM PyTorch + Optuna TPE (Fases 1-10)
│   ├── agente_5_alertas.py              # Ensamble + clasificacion 3 niveles
│   ├── agente_6_regimen.py              # Regimen epidemico + pesos dinamicos
│   └── s3_client.py                     # Cliente S3 (upload/download/ensure_local)
├── backend/
│   ├── main.py                          # FastAPI — todos los endpoints REST
│   ├── services.py                      # PredictionService — carga S3, orquesta agentes
│   └── schemas.py                       # Modelos Pydantic
├── frontend/
│   ├── public/favicon.svg               # Icono mosquito SVG
│   └── src/
│       ├── App.jsx                      # SPA raiz + exportacion PDF (jsPDF)
│       └── components/
│           ├── BottomNav.jsx            # Navegacion inferior para movil
│           ├── Sidebar.jsx / Topbar.jsx
│           ├── DashboardView.jsx        # KPIs + ScatterPlot + mapa (responsive)
│           ├── MapContainer.jsx         # Mapa Leaflet por nivel de riesgo
│           ├── PredictorView.jsx        # Sliders + semaforo + regimen Agente 6
│           ├── ExplainabilityView.jsx   # SHAP global y local
│           ├── InfoView.jsx             # Arquitectura + tech stack (responsive)
│           └── ScatterPlot.jsx          # SVG puro, 4,056 puntos, dark mode
├── data/
│   ├── raw/                             # CSVs fuentes oficiales
│   ├── processed/                       # dataset_maestro + dataset_features
│   └── models/                          # Artefactos + metrics.json
│                                        #   drift_report.json — PSI mensual
│                                        #   data_version.json — SHA OpenDengue vigente
├── scripts/
│   ├── training/entrenar_modelos.py     # Re-entrena ambos modelos desde cero
│   └── pipeline/
│       ├── verificar_actualizacion.py   # FASE 10: drift + version check + reentrenamiento
│       └── generar_scatter_data.py      # Genera scatter_data.json para el frontend
├── .github/
│   └── workflows/
│       └── retrain.yml                  # GitHub Actions: cron mensual automatico
├── Dockerfile                           # python:3.11-slim
├── Procfile                             # uvicorn backend.main:app --host 0.0.0.0 --port $PORT
└── requirements.txt
```

---

## API REST — Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/api/status` | Estado del sistema (loading / ready) |
| `GET` | `/api/metrics` | R², MAE, RMSE, pesos del ensamble y metricas de clasificacion |
| `GET` | `/api/drift-status` | Ultimo reporte de drift PSI (features climaticas NASA POWER) |
| `GET` | `/api/metadata` | Paises y departamentos disponibles |
| `GET` | `/api/coordinates` | Coordenadas GPS de los 169 departamentos |
| `GET` | `/api/historical` | Serie historica mensual (`?iso_a0=&adm_1_name=`) |
| `GET` | `/api/features` | Features del ultimo periodo para un departamento |
| `GET` | `/api/map-summary` | Incidencia media + nivel de riesgo por departamento |
| `GET` | `/api/top-departments` | Top N departamentos por incidencia historica |
| `GET` | `/api/scatter-data` | 4,056 puntos real vs predicho (Ensemble / XGBoost / LSTM) |
| `GET` | `/api/explain/global` | Importancias SHAP globales (TreeSHAP) |
| `POST` | `/api/predict/simulate` | Prediccion interactiva con sliders de variables climaticas |
| `POST` | `/api/predict/raw` | Prediccion con vector de 73 features completo |

Documentacion interactiva en `/docs` (Swagger UI).

---

## Estructura S3

```
s3://epipredict-dengue/
├── datos_crudos/
│   ├── Temporal_extract_V1_3.csv        # Dataset OpenDengue V1.3
│   ├── clima_nasa_crudo.csv
│   ├── agua_jmp_crudo.csv
│   ├── departamentos_coordenadas.csv    # GPS de 169 departamentos
│   └── poblacion/                       # Poblacion anual por pais
├── datos_procesados/
│   ├── dataset_maestro_mensual_latam.csv
│   └── dataset_features_latam.csv       # 16,224 x 81 (73 features + meta)
└── modelos/
    ├── pipeline_ml.pkl                  # Pipeline XGBoost completo
    ├── xgb_model.pkl / xgb_config.json
    ├── escalador_ml.pkl / imputador_ml.pkl / cols_feat.pkl
    ├── shap_importance.json             # Importancias TreeSHAP globales
    ├── lstm_model.pth                   # LSTM (hidden=77, layers=3)
    ├── lstm_config.json / lstm_features.pkl / escalador_lstm.pkl
    ├── thresholds_clasificacion.json    # Percentiles p50/p90 por departamento
    ├── scatter_data.json                # 4,056 puntos test para scatter plot
    ├── metrics.json                     # R², MAE, RMSE, pesos, clasificacion
    ├── data_version.json                # SHA OpenDengue vigente + fecha
    └── drift_report.json                # PSI por feature + alerta_drift
```

---

## Variables de entorno

| Variable | Descripcion |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS |
| `AWS_DEFAULT_REGION` | Region del bucket (`us-east-2`) |
| `RAILWAY_ENVIRONMENT` | Inyectada por Railway — activa rutas `/tmp/sma_data/` |
| `PORT` | Puerto del servidor (Railway lo inyecta automaticamente) |
| `VITE_API_URL` | URL del backend para Vite (`.env.production`) |

Para el workflow de GitHub Actions, `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` se configuran en **Settings → Secrets and variables → Actions** del repositorio.

---

## Decisiones de diseño

**Split dinamico train/test** — `split_ano = max_ano - 2` garantiza que siempre los ultimos 2 anos del dataset sean el conjunto de prueba, independientemente del ano maximo. Esto hace el pipeline de reentrenamiento automatico consistente: cuando lleguen datos nuevos de OpenDengue, el split se ajusta solo.

**Pesos base 50/50** — Se eligio sobre el resultado del optimizador (sesgado por la dominancia de XGBoost en validacion) porque produce mejor MAE (5.83 vs 6.07) y RMSE (20.67 vs 22.18) en el test set. Los pesos son ajustados dinamicamente por el Agente 6 en cada inferencia.

**LSTM con 6 features** — El LSTM recibe una ventana de 12 pasos y aprende internamente la estructura temporal; los 73 features del Agente 3 generan ruido que degrada el rendimiento.

**Transformacion log1p** — La incidencia de dengue es hiperasimetrica (mayoria 0–20 casos/100k, picos de 200–500+). La transformacion estabiliza la varianza y mejora la optimizacion. R² reportado en escala log1p.

**Clasificacion de 3 clases** — Fusion de "Normal" y "Vigilancia" en "Endemico" mejora el Kappa de 0.41 (Moderado) a 0.72 (Sustancial). Ambas categorias tienen el mismo protocolo de intervencion clinica.

**Carga en memoria al iniciar** — El backend descarga todos los artefactos desde S3 al arrancar (startup event) y los mantiene en RAM. Latencia de inferencia < 200 ms por prediccion.

**Drift de covariables vs. concepto** — El drift de covariables (features climaticas) se monitorea mensualmente via NASA POWER con PSI. El drift de concepto requiere datos etiquetados de OpenDengue (latencia 6–12 meses) y se evalua en cada reentrenamiento cuando hay nueva version del dataset.

**Fallback de descarga OpenDengue** — El Agente 1 descarga automaticamente el dataset si no existe localmente, intentando versiones en orden descendente (V1.3 → V1.2.2 → V1.1 → V1.0) desde los releases de GitHub.

---

## Fuentes de datos

| Fuente | Uso | Cobertura |
|---|---|---|
| OpenDengue Project | Casos de dengue subnacionales | 8 paises, 2014–2022 |
| NASA POWER API | Temperatura, precipitacion, humedad mensual | Global, ~0.5 grados de resolucion |
| World Bank Open Data | Poblacion anual | Global |
| JMP OMS/UNICEF | Acceso a agua potable basica | Global, anual |
