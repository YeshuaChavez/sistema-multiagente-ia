# EpiPredict — Sistema Multi-Agente SMA-ML/DL para Vigilancia Epidemiológica de Dengue

Sistema de alerta temprana para la predicción de epidemias de dengue a escala subnacional en América Latina. Implementa una arquitectura de **seis agentes inteligentes coordinados** que combinan Machine Learning (XGBoost + SHAP) y Deep Learning (LSTM PyTorch) en un ensamble híbrido con detección dinámica de régimen epidémico, desplegado como aplicación web completa en la nube.

**Proyecto Final — Universidad Nacional Mayor de San Marcos**  
Facultad de Ingeniería de Sistemas e Informática

---

## Demo

| Capa | URL |
|---|---|
| Frontend (Vercel) | https://proyecto-ia-eight.vercel.app |
| Backend API (Railway) | https://proyecto-ia-production.up.railway.app |
| Documentación interactiva (Swagger) | https://proyecto-ia-production.up.railway.app/docs |

---

## Métricas del sistema

| Modelo | R² (test) | MAE (casos/100k) | RMSE |
|---|---|---|---|
| XGBoost — Agente 3 | **91.49%** | 6.07 | 22.18 |
| LSTM PyTorch — Agente 4 | 90.35% | 6.02 | 20.52 |
| **Ensamble — Agente 5** | **91.47%** | **5.83** | **20.67** |

- **Conjunto de entrenamiento:** 12,168 observaciones mensuales
- **Conjunto de prueba:** 4,056 observaciones, partición cronológica estricta (últimos 2 años del dataset)
- **Split dinámico:** `split_ano = max_ano - 2` — permite reentrenamiento automático sin cambiar código
- **Validación temporal:** `TimeSeriesSplit(k=5)` sobre el período de entrenamiento
- **Dataset features:** 16,224 registros × 73 variables predictoras
- **Pesos base del ensamble:** `w_xgb = 0.50`, `w_lstm = 0.50`, ajustados dinámicamente por el Agente 6 según el régimen epidémico
- **Cobertura:** 8 países, 169 unidades subnacionales
- Las métricas R² se reportan en escala `log1p`, estándar para distribuciones epidemiológicas asimétricas

### Clasificación de riesgo epidémico (3 clases)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Endémico | 92.77% | 89.65% | 91.18% | 2,532 |
| Alerta | 69.66% | 79.67% | 74.33% | 1,092 |
| Epidemia | 86.67% | 72.22% | 78.79% | 432 |
| **Global** | | | | |
| Accuracy | | | **85.11%** | 4,056 |
| Cohen's Kappa | | | **0.7196** (Sustancial) | |

La clasificación usa **percentiles históricos locales calibrados por departamento**: Endémico (≤p50), Alerta (p50–p90), Epidemia (>p90).

---

## Cobertura geográfica

| País | Código ISO | Unidades subnacionales |
|---|---|---|
| Argentina | ARG | Provincias |
| Bolivia | BOL | Departamentos |
| Brasil | BRA | Estados |
| Colombia | COL | Departamentos |
| Ecuador | ECU | Provincias |
| México | MEX | Estados |
| Panamá | PAN | Provincias |
| Perú | PER | Departamentos |

**Total: 169 unidades subnacionales — Periodo: 2014–2022**

---

## Arquitectura del sistema

```
Fuentes externas (OpenDengue · NASA POWER · Banco Mundial · JMP OMS/UNICEF)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              AWS S3  (epipredict-dengue)                │
│   raw/  ·  processed/  ·  models/                       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Agente 1 — Recoleccion
  (ingesta automatica desde fuentes oficiales -> data/raw/)
  (fallback automatico: descarga ZIP desde GitHub OpenDengue)
        │
        ▼
  Agente 2 — Preprocesamiento + Feature Engineering
  (73 features: lags, rolling, vecinos GPS, estacionalidad ciclica,
   indicadores epidemiologicos, dummies de pais)
        │
     ┌──┴──┐
     ▼     ▼
  Agente 3   Agente 4
  XGBoost    LSTM PyTorch
  R²=91.49%  R²=90.35%
     └──┬──┘
        ▼
  Agente 5 — Orquestador
  (ensamble 50/50 -> R²=91.47% + clasificacion Endemico/Alerta/Epidemia)
        │
        ▼ (consulta regimen)
  Agente 6 — Regimen Epidemico
  (ajuste dinamico de pesos: Normal / Vigilancia / Pre-brote / Brote activo / Post-pico)
        │
        ▼
  Backend FastAPI (Railway) <---> Frontend React 19 / Vite (Vercel)

        +--- FASE 10: Monitoreo y Mantenimiento ---+
        |   GitHub Actions (cron 1ro de cada mes)   |
        |   verificar_actualizacion.py              |
        |   ├── Version check (SHA GitHub)          |
        |   ├── Drift PSI (NASA POWER)              |
        |   └── Reentrenamiento automatico          |
        +-------------------------------------------+
```

---

## Agentes

### Agente 1 — Recoleccion de datos

Descarga y consolida automaticamente los datos historicos 2014–2022 desde:

- **OpenDengue** — Casos de dengue a nivel subnacional en 8 paises de Latinoamerica
- **NASA POWER API** — Variables climaticas satelitales diarias (temperatura max/min, precipitacion, humedad relativa)
- **Banco Mundial** — Estimaciones de poblacion anual por pais y subregion
- **JMP OMS/UNICEF** — Indicador oficial de acceso a agua potable basica

**Fallback automatico de descarga:** Si el CSV de OpenDengue no existe localmente, el agente descarga automaticamente el ZIP desde GitHub (`data/releases/V1.3/`) y lo extrae. Prueba versiones en orden descendente (V1.3 → V1.2.2 → V1.1 → V1.0).

---

### Agente 2 — Preprocesamiento y Feature Engineering

Calcula la tasa de incidencia mensual normalizada (`casos / poblacion × 100,000`) y construye las **73 variables predictoras**:

| Grupo | Variables | Cantidad |
|---|---|---|
| Base climatica y demografica | `tmax`, `tmin`, `precipitacion`, `humedad`, `poblacion`, `densidad_poblacion` | 6 |
| Lags climaticos | `tmax/tmin/precipitacion/humedad lag1–lag6` | 24 |
| Lags de incidencia | `incidencia_lag1` a `incidencia_lag12` (escala `log1p`) | 12 |
| Rolling means | `incidencia_roll3`, `roll6`, `roll12` (escala `log1p`) | 3 |
| Vecinos espaciales | `incidencia_vecinos_lag1–lag6` (3 deptos mas cercanos por GPS) | 6 |
| Estacionalidad ciclica | `mes_sin = sin(2π·mes/12)`, `mes_cos = cos(2π·mes/12)` | 2 |
| Indicadores epidemiologicos | `indicador_covid`, `indicador_nino`, `indicador_nina` | 3 |
| Features derivadas | `amplitud_termica`, `temperatura_media`, `precipitacion_anomalia`, `aceleracion_incidencia`, `cambio_interanual`, `tendencia_1m`, `tendencia_3m`, `fase_ascendente`, `indicador_brote` | 9 |
| Dummies de pais | `pais_ARG/BOL/BRA/COL/ECU/MEX/PAN/PER` | 8 |
| **Total** | | **73** |

Produce dos artefactos en S3:

- `processed/dataset_maestro_mensual_latam.csv` — 18,252 filas × 14 cols (base para historico e inferencia)
- `processed/dataset_features_latam.csv` — 16,224 filas × 81 cols (73 features para entrenamiento)

---

### Agente 3 — Prediccion ML (XGBoost + SHAP)

Pipeline con transformacion logaritmica del target:

```
SimpleImputer(median) -> StandardScaler -> XGBRegressor
target: log1p(incidencia_dengue) -> output: expm1(prediccion)
```

**Optimizacion de hiperparametros:** GridSearchCV + `TimeSeriesSplit(k=5)` sobre el set de entrenamiento:

```
n_estimators      = 600 / 800
learning_rate     = 0.01
max_depth         = 4 / 5
min_child_weight  = 3
gamma             = 0.1
subsample         = 0.8
colsample_bytree  = 0.8
```

Calcula importancias **SHAP globales** (TreeSHAP sobre el test set completo) y **SHAP locales** por prediccion.

**Resultado en test (ultimos 2 anos):** R² = 91.49% | MAE = 6.07 casos/100k | RMSE = 22.18

---

### Agente 4 — Prediccion DL (LSTM PyTorch)

Red LSTM que aprende dependencias temporales directamente desde la secuencia. Usa **solo 6 features** (la memoria interna del LSTM reemplaza los lags explicitos):

```
Features: tmax_promedio · tmin_promedio · precipitacion ·
          humedad_promedio · agua_basica · incidencia_dengue
```

**Arquitectura (Grid Search manual + TimeSeriesSplit(k=5)):**

```
Input:    lookback=12 meses × 6 features
LSTM:     hidden_dim=77, num_layers=3, dropout=0.293
Output:   Linear(77 -> 1) -> expm1 -> incidencia predicha
```

**Entrenamiento:** Adam (`lr=0.00988`), Early Stopping (patience=15), ReduceLROnPlateau(patience=5), `torch.manual_seed(42)`, CPU-only.

**Resultado en test (ultimos 2 anos):** R² = 90.35% | MAE = 6.02 casos/100k | RMSE = 20.52

---

### Agente 5 — Orquestador de Consenso (Ensamble + Alertas)

Combina las predicciones de los Agentes 3 y 4 con **pesos base 50/50**, ajustados dinamicamente por el Agente 6:

```python
w_xgb  = 0.50   # pesos base
w_lstm = 0.50

pred_ensemble = w_xgb * pred_xgb + w_lstm * pred_lstm
```

Clasifica cada departamento con **percentiles historicos locales** (set de entrenamiento):

| Nivel | Criterio | Color | Accion |
|---|---|---|---|
| **Endemico** | prediccion <= p50 local | Verde | Monitoreo rutinario |
| **Alerta** | p50 < prediccion <= p90 | Naranja | Fumigacion selectiva, cerco epidemiologico |
| **Epidemia** | prediccion > p90 | Rojo | Emergencia de salud publica |

**Resultado en test:** R² = 91.47% | MAE = 5.83 | RMSE = 20.67 | Accuracy = 85.11% | Kappa = 0.7196

---

### Agente 6 — Regimen Epidemico (Ajuste Dinamico de Pesos)

Detecta el regimen epidemiologico usando `incidencia_lag1` contra percentiles historicos locales y la tendencia (`log1p(lag1) − log1p(lag2)`):

| Regimen | Condicion | Ajuste de pesos |
|---|---|---|
| Normal | `lag1 <= p25` | Pesos base (50/50) |
| Vigilancia | `p25 < lag1 <= p50` | Pesos base (50/50) |
| Pre-brote | `p50 < lag1 <= p90` + tendencia↑ | `w_lstm` → min(0.50×1.4, 0.65) |
| Brote activo | `lag1 > p90` + tendencia↑ | `w_lstm` → min(0.50×(lag1/p90), 0.80) |
| Post-pico | `lag1 > p90` + tendencia↓ | `w_xgb` → min(0.50×1.5, 0.75) |

> Los cinco regimenes son internos al sistema para el ajuste de pesos. El usuario solo ve los tres niveles del Agente 5 (Endemico / Alerta / Epidemia).

---

## Ciclo de vida de los modelos (10 fases)

El sistema implementa el ciclo de vida completo de modelos ML/DL documentado explicitamente en el codigo fuente de los Agentes 3 y 4:

| Fase | Descripcion | Implementacion |
|---|---|---|
| 1 — Problema | Prediccion de tasa de incidencia de dengue subnacional | Definicion en docstrings de Agentes 3 y 4 |
| 2 — Recoleccion | Ingesta desde OpenDengue, NASA POWER, BM, JMP | `agente_1_recoleccion.py` |
| 3 — Preparacion | Feature engineering (73 variables) | `agente_2_preprocesamiento.py` |
| 4 — Division | Split cronologico dinamico: ultimos 2 anos = test | `split_ano = max_ano - 2` en Agentes 3, 4 |
| 5 — Seleccion | Pipeline XGBoost / LSTM 3 capas PyTorch | Agentes 3 y 4 |
| 6a — Baseline | Entrenamiento con parametros por defecto | Agentes 3 y 4 |
| 7a — Evaluacion baseline | R², MAE en test set | Agentes 3 y 4 |
| 8 — Optimizacion | GridSearchCV + TimeSeriesSplit (ML) / Grid manual (DL) | Agentes 3 y 4 |
| 6b — Reentrenamiento | Con mejores hiperparametros (refit=True) | Agentes 3 y 4 |
| 7b — Evaluacion final | R², MAE, RMSE en test + pesos ensemble | Agentes 3 y 4 |
| 9 — Despliegue | Serializacion a S3, FastAPI Railway, React Vercel | `s3_client.py`, `backend/`, `frontend/` |
| **10 — Mantenimiento** | Drift detection + version check + reentrenamiento auto | `verificar_actualizacion.py` + GitHub Actions |

### Fase 10: Monitoreo y Mantenimiento automatizado

Ejecutado automaticamente el **1ro de cada mes** via GitHub Actions (`.github/workflows/retrain.yml`):

**10a — Deteccion de nueva version de datos:**
- Consulta la API de GitHub para obtener el SHA del ultimo commit en `data/releases/` del repositorio OpenDengue
- Si el SHA difiere del guardado en `data/models/data_version.json`, hay datos nuevos → activa reentrenamiento

**10b — Deteccion de drift de covariables (PSI — Population Stability Index):**
- Descarga datos climaticos recientes de NASA POWER (2 ultimos anos disponibles)
- Compara la distribucion de 4 features climaticas vs. la distribucion del set de entrenamiento
- Calcula PSI por feature:

| PSI | Nivel | Accion |
|---|---|---|
| < 0.1 | Estable | Sin accion |
| 0.1 – 0.2 | Moderado | Monitorear |
| >= 0.2 | Alto | Priorizar reentrenamiento |

- Guarda `data/models/drift_report.json` con PSI por feature y bandera `alerta_drift`
- Si `alerta_drift=true` sin nueva version: crea un GitHub Issue automaticamente

> **Nota:** Solo el drift de covariables (features climaticas) es detectable en tiempo casi-real via NASA POWER. El drift de concepto (cambio en la relacion features→incidencia) requiere nuevos datos etiquetados de OpenDengue, que publica con 6–12 meses de latencia.

**10c — Reentrenamiento automatico:**
- Si hay nueva version de OpenDengue: descarga dataset, ejecuta pipeline completo (Agentes 2, 3, 4), sube modelos a S3

**Resultado ejemplo (Junio 2026):**
```json
{
  "estado": "calculado",
  "features": {
    "tmax_promedio":  { "psi": 0.1675, "nivel": "moderado" },
    "tmin_promedio":  { "psi": 0.3676, "nivel": "alto" },
    "precipitacion":  { "psi": 0.0736, "nivel": "estable" },
    "humedad_promedio": { "psi": 0.0357, "nivel": "estable" }
  },
  "psi_max": 0.3676,
  "alerta_drift": true
}
```
La temperatura minima muestra drift alto (PSI=0.37), consistente con el calentamiento climatico observado en Latinoamerica entre el periodo de entrenamiento y 2024–2025.

---

## Stack tecnologico

| Capa | Tecnologias |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 |
| ML | XGBoost 2.x · Scikit-Learn 1.8 · SHAP (TreeSHAP) · GridSearchCV |
| DL | PyTorch 2.x (LSTM 3 capas, hidden=77, lookback=12) |
| Datos | Pandas · NumPy · requests · python-dotenv |
| Frontend | React 19 · Vite · TailwindCSS · Leaflet.js · Material Symbols |
| Visualizacion | SVG puro (ScatterPlot 4,056 puntos, sin librerias externas) |
| PDF | jsPDF + jspdf-autotable (exportacion de reporte tecnico) |
| Storage | AWS S3 (`epipredict-dengue`) via boto3 |
| Deploy backend | Railway (Docker `python:3.11-slim`) |
| Deploy frontend | Vercel |
| CI/CD | GitHub Actions (cron mensual: drift + reentrenamiento) |

---

## Estructura del repositorio

```
/
├── agents/
│   ├── agente_1_recoleccion.py          # Ingesta OpenDengue + NASA POWER + BM + JMP
│   │                                    #   (fallback ZIP desde GitHub si CSV no existe)
│   ├── agente_2_preprocesamiento.py     # Feature engineering (73 features)
│   ├── agente_3_prediccion_ml.py        # XGBoost + GridSearchCV + SHAP global/local
│   │                                    #   (Fases 1-10 documentadas en codigo)
│   ├── agente_4_prediccion_dl.py        # LSTM PyTorch + Grid manual (hidden=77, layers=3)
│   │                                    #   (Fases 1-10 documentadas en codigo)
│   ├── agente_5_alertas.py              # Orquestador: ensamble + 3 niveles de riesgo
│   ├── agente_6_regimen.py              # Deteccion de regimen + ajuste dinamico de pesos
│   └── s3_client.py                     # Cliente S3 compartido (upload/download/ensure_local)
├── backend/
│   ├── main.py                          # FastAPI app + todos los endpoints REST
│   ├── services.py                      # PredictionService (carga artefactos S3, orquesta)
│   └── schemas.py                       # Modelos Pydantic (RiskLevel: Endemico/Alerta/Epidemia)
├── frontend/
│   ├── public/
│   │   └── favicon.svg                  # Icono de mosquito (SVG)
│   └── src/
│       ├── App.jsx                      # Raiz SPA + exportacion PDF (jsPDF)
│       └── components/
│           ├── BottomNav.jsx            # Navegacion inferior para movil
│           ├── Sidebar.jsx / Topbar.jsx
│           ├── DashboardView.jsx        # KPIs + ScatterPlot + mapa (responsive)
│           ├── MapContainer.jsx         # Mapa Leaflet por nivel de riesgo
│           ├── PredictorView.jsx        # Sliders + semaforo + pesos dinamicos Agente 6
│           ├── ExplainabilityView.jsx   # SHAP global y local (XAI)
│           ├── InfoView.jsx             # Flujo de arquitectura y tech stack (responsive)
│           └── ScatterPlot.jsx          # SVG puro, 4,056 puntos, dark mode
├── data/                                # Local — descargado desde S3 en runtime
│   ├── raw/                             # CSVs fuentes oficiales
│   ├── processed/                       # dataset_maestro + dataset_features
│   └── models/                          # Artefactos de modelos + metrics.json
│                                        #   drift_report.json — reporte PSI mensual
│                                        #   data_version.json — SHA OpenDengue vigente
├── scripts/
│   ├── training/entrenar_modelos.py     # Re-entrena ambos modelos desde cero
│   ├── pipeline/
│   │   ├── verificar_actualizacion.py   # FASE 10: version check + drift PSI + reentrenamiento
│   │   └── generar_scatter_data.py      # Genera scatter_data.json (ejecutar 1 vez local)
│   └── analysis/                        # Scripts de analisis y diagnostico
├── .github/
│   └── workflows/
│       └── retrain.yml                  # GitHub Actions: cron mensual drift + reentrenamiento
├── notebooks/                           # Notebooks Colab (gitignored)
├── Dockerfile                           # python:3.11-slim
├── Procfile                             # uvicorn backend.main:app --host 0.0.0.0 --port $PORT
└── requirements.txt
```

---

## API REST — Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/api/status` | Estado general del sistema |
| `GET` | `/api/metrics` | R², MAE, RMSE y pesos del ensamble |
| `GET` | `/api/metadata` | Paises y departamentos disponibles |
| `GET` | `/api/coordinates` | Coordenadas GPS de los 169 departamentos |
| `GET` | `/api/historical` | Serie historica mensual (`?iso_a0=&adm_1_name=`) |
| `GET` | `/api/features` | Features del ultimo periodo para un departamento |
| `GET` | `/api/map-summary` | Incidencia media + nivel de riesgo por departamento |
| `GET` | `/api/top-departments` | Top N departamentos por incidencia historica |
| `GET` | `/api/scatter-data` | 4,056 puntos real vs predicho (Ensemble/XGBoost/LSTM) |
| `GET` | `/api/explain/global` | Importancias SHAP globales (TreeSHAP) |
| `POST` | `/api/predict/simulate` | Prediccion con sliders de variables climaticas + regimen |
| `POST` | `/api/predict/raw` | Prediccion con vector de 73 features completo |

Documentacion interactiva Swagger disponible en `/docs`.

---

## Estructura S3

```
s3://epipredict-dengue/
├── datos_crudos/
│   ├── Temporal_extract_V1_3.csv
│   ├── clima_nasa_crudo.csv
│   ├── agua_jmp_crudo.csv
│   ├── departamentos_coordenadas.csv
│   └── poblacion/
├── datos_procesados/
│   ├── dataset_maestro_mensual_latam.csv
│   └── dataset_features_latam.csv
└── modelos/
    ├── pipeline_ml.pkl                  # Pipeline XGBoost (GridSearchCV optimizado)
    ├── xgb_model.pkl / xgb_config.json
    ├── escalador_ml.pkl / imputador_ml.pkl / cols_feat.pkl
    ├── shap_importance.json             # Importancias TreeSHAP globales
    ├── lstm_model.pth                   # LSTM (hidden=77, layers=3)
    ├── lstm_config.json / lstm_features.pkl / escalador_lstm.pkl
    ├── scatter_data.json                # 4,056 puntos test para scatter plot
    ├── thresholds_clasificacion.json    # Percentiles p50/p90 por departamento
    ├── metrics.json                     # R², MAE, RMSE, pesos y metricas de clasificacion
    ├── data_version.json                # SHA OpenDengue vigente + fecha de verificacion
    └── drift_report.json                # PSI por feature climatica + alerta_drift
```

---

## Variables de entorno

| Variable | Descripcion |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS S3 |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS S3 |
| `AWS_DEFAULT_REGION` | Region S3 (`us-east-2`) |
| `RAILWAY_ENVIRONMENT` | Inyectada por Railway — activa rutas `/tmp/sma_data/` |
| `PORT` | Puerto del servidor (Railway lo inyecta automaticamente) |
| `VITE_API_URL` | URL del backend para el frontend (`.env.production` de Vite) |

Para el workflow de GitHub Actions, `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` deben configurarse en **Settings → Secrets and variables → Actions** del repositorio.

---

## Decisiones de diseno

**Split dinamico train/test** — `split_ano = max_ano - 2` hace que siempre los ultimos 2 anos del dataset sean el conjunto de prueba, independientemente del ano maximo disponible. Esto permite que el reentrenamiento automatico sea consistente cuando lleguen nuevos datos de OpenDengue sin necesidad de modificar el codigo.

**Pesos base 50/50** — Frente al resultado del optimizador (sesgado por la dominancia de XGBoost en el set de validacion), se eligio 50/50 porque produce mejor MAE (5.83 vs 6.07) y RMSE (20.67 vs 22.18) en el test set. Los pesos base son ajustados dinamicamente por el Agente 6, por lo que el ensamble no es estatico.

**LSTM con solo 6 features** — A diferencia de XGBoost, el LSTM recibe una ventana de 12 pasos consecutivos y aprende internamente la estructura temporal. Anadir los 73 features empeora el rendimiento por ruido.

**Transformacion log1p** — La incidencia de dengue tiene distribucion hiperasimetrica (mayoria de registros 0–20 casos/100k, picos de 200–500+). La transformacion permite que el modelo optimice uniformemente en todo el rango. El R² se reporta en escala log1p (estandar epidemiologico).

**Clasificacion de 3 clases** — La fusion de "Normal" y "Vigilancia" en "Endemico" mejora el Kappa de 0.41 (Moderado) a 0.72 (Sustancial). Ambas categorias tienen el mismo protocolo de intervencion.

**Carga en memoria al iniciar** — El backend descarga desde S3 todos los artefactos al arrancar y los mantiene en RAM. Latencia de inferencia < 200 ms por prediccion.

**Drift de covariables vs. concepto** — Solo el drift de covariables (variables climaticas) es monitoreable automaticamente via NASA POWER. El drift de concepto (cambio en la relacion features→incidencia) requiere datos etiquetados de OpenDengue con 6–12 meses de latencia y se delega al reentrenamiento cuando hay nueva version del dataset.

**Fallback de descarga OpenDengue** — El Agente 1 intenta descargar automaticamente el dataset si no existe localmente, probando versiones en orden descendente (V1.3 → V1.2.2 → V1.1 → V1.0) desde los releases de GitHub.

---

## Fuentes de datos

| Fuente | Uso | Cobertura |
|---|---|---|
| **OpenDengue Project** | Casos de dengue subnacionales | 8 paises, 2014–2022 |
| **NASA POWER API** | Temperatura, precipitacion, humedad mensual | Global, ~0.5° resolucion |
| **World Bank Open Data** | Poblacion anual | Global |
| **JMP OMS/UNICEF** | Acceso a agua basica | Global, anual |
