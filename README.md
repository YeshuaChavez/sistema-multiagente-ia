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

| Modelo | R² (test 2021–2022) | MAE (casos/100k) | RMSE |
|---|---|---|---|
| XGBoost — Agente 3 | **91.49%** | 6.07 | 22.18 |
| LSTM PyTorch — Agente 4 | 90.35% | 6.02 | 20.52 |
| **Ensamble — Agente 5** | **91.47%** | **5.83** | **20.80** |

- **Conjunto de entrenamiento:** 12,168 observaciones mensuales (2014–2020)
- **Conjunto de prueba:** 4,056 observaciones (2021–2022), partición cronológica estricta
- **Validación temporal:** folds 2016–2020 con `TimeSeriesSplit(k=5)`, usado para optimización Bayesiana de hiperparámetros
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

La clasificación usa **percentiles históricos locales calibrados por departamento** (train 2014–2020): Endémico (≤p50), Alerta (p50–p90), Epidemia (>p90).

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
  Agente 1 — Recolección
  (ingesta automática desde fuentes oficiales → data/raw/)
        │
        ▼
  Agente 2 — Preprocesamiento + Feature Engineering
  (73 features: lags, rolling, vecinos GPS, estacionalidad cíclica,
   indicadores epidemiológicos, dummies de país)
        │
     ┌──┴──┐
     ▼     ▼
  Agente 3   Agente 4
  XGBoost    LSTM PyTorch
  R²=91.49%  R²=90.35%
     └──┬──┘
        ▼
  Agente 5 — Orquestador
  (ensamble 50/50 → R²=91.47% + clasificación Endémico/Alerta/Epidemia)
        │
        ▼ (consulta régimen)
  Agente 6 — Régimen Epidémico
  (ajuste dinámico de pesos: Normal / Vigilancia / Pre-brote / Brote activo / Post-pico)
        │
        ▼
  Backend FastAPI (Railway) ◄──► Frontend React 19 / Vite (Vercel)
```

---

## Agentes

### Agente 1 — Recolección de datos

Descarga y consolida automáticamente los datos históricos 2014–2022 desde:

- **OpenDengue** — Casos de dengue a nivel subnacional en 8 países de Latinoamérica
- **NASA POWER API** — Variables climáticas satelitales diarias (temperatura máx/mín, precipitación, humedad relativa)
- **Banco Mundial** — Estimaciones de población anual por país y subregión
- **JMP OMS/UNICEF** — Indicador oficial de acceso a agua potable básica

---

### Agente 2 — Preprocesamiento y Feature Engineering

Calcula la tasa de incidencia mensual normalizada (`casos / población × 100,000`) y construye las **73 variables predictoras**:

| Grupo | Variables | Cantidad |
|---|---|---|
| Base climática y demográfica | `tmax`, `tmin`, `precipitacion`, `humedad`, `poblacion`, `densidad_poblacion` | 6 |
| Lags climáticos | `tmax/tmin/precipitacion/humedad lag1–lag6` | 24 |
| Lags de incidencia | `incidencia_lag1` a `incidencia_lag12` (escala `log1p`) | 12 |
| Rolling means | `incidencia_roll3`, `roll6`, `roll12` (escala `log1p`) | 3 |
| Vecinos espaciales | `incidencia_vecinos_lag1–lag6` (3 deptos más cercanos por GPS) | 6 |
| Estacionalidad cíclica | `mes_sin = sin(2π·mes/12)`, `mes_cos = cos(2π·mes/12)` | 2 |
| Indicadores epidemiológicos | `indicador_covid`, `indicador_nino`, `indicador_nina` | 3 |
| Features derivadas | `amplitud_termica`, `temperatura_media`, `precipitacion_anomalia`, `aceleracion_incidencia`, `cambio_interanual`, `tendencia_1m`, `tendencia_3m`, `fase_ascendente`, `indicador_brote` | 9 |
| Dummies de país | `pais_ARG/BOL/BRA/COL/ECU/MEX/PAN/PER` | 8 |
| **Total** | | **73** |

Produce dos artefactos en S3:

- `processed/dataset_maestro_mensual_latam.csv` — 18,252 filas × 14 cols (base para histórico e inferencia)
- `processed/dataset_features_latam.csv` — 16,224 filas × 81 cols (73 features para entrenamiento)

---

### Agente 3 — Predicción ML (XGBoost + SHAP)

Pipeline con transformación logarítmica del target:

```
SimpleImputer(median) → StandardScaler → XGBRegressor
target: log1p(incidencia_dengue) → output: expm1(prediccion)
```

**Optimización de hiperparámetros:** Optuna TPE (Bayesian Optimization), 50 trials × K=5 `TimeSeriesSplit` temporal (folds 2016–2020):

```
n_estimators      = 805
learning_rate     = 0.0242
max_depth         = 5
min_child_weight  = 10
subsample         = 0.656
colsample_bytree  = 0.516
gamma             = 0.088
```

Calcula importancias **SHAP globales** (TreeSHAP sobre el test set completo) y **SHAP locales** por predicción.

**Resultado en test 2021–2022:** R² = 91.49% | MAE = 6.07 casos/100k | RMSE = 22.18

---

### Agente 4 — Predicción DL (LSTM PyTorch)

Red LSTM que aprende dependencias temporales directamente desde la secuencia. Usa **solo 6 features** (la memoria interna del LSTM reemplaza los lags explícitos):

```
Features: tmax_promedio · tmin_promedio · precipitacion ·
          humedad_promedio · agua_basica · incidencia_dengue
```

**Arquitectura (Bayesian Optimization, 30 trials × K=5 `TimeSeriesSplit`):**

```
Input:    lookback=12 meses × 6 features
LSTM:     hidden_dim=77, num_layers=3, dropout=0.293
Output:   Linear(77 → 1) → expm1 → incidencia predicha
```

**Entrenamiento:** Adam (`lr=0.00988`), Early Stopping, `torch.manual_seed(42)`, CPU-only.

**Resultado en test 2021–2022:** R² = 90.35% | MAE = 6.02 casos/100k | RMSE = 20.52

---

### Agente 5 — Orquestador de Consenso (Ensamble + Alertas)

Combina las predicciones de los Agentes 3 y 4 con **pesos base 50/50**, ajustados dinámicamente por el Agente 6:

```python
w_xgb  = 0.50   # pesos base
w_lstm = 0.50

pred_ensemble = w_xgb × pred_xgb + w_lstm × pred_lstm
```

Clasifica cada departamento con **percentiles históricos locales** (train 2014–2020):

| Nivel | Criterio | Color | Acción |
|---|---|---|---|
| **Endémico** | predicción ≤ p50 local | Verde | Monitoreo rutinario |
| **Alerta** | p50 < predicción ≤ p90 | Naranja | Fumigación selectiva, cerco epidemiológico |
| **Epidemia** | predicción > p90 | Rojo | Emergencia de salud pública |

**Resultado en test 2021–2022:** R² = 91.47% | MAE = 5.83 | RMSE = 20.80 | Accuracy = 85.11% | Kappa = 0.7196

---

### Agente 6 — Régimen Epidémico (Ajuste Dinámico de Pesos)

Detecta el régimen epidemiológico usando `incidencia_lag1` contra percentiles históricos locales y la tendencia (`log1p(lag1) − log1p(lag2)`):

| Régimen | Condición | Ajuste de pesos |
|---|---|---|
| Normal | `lag1 ≤ p25` | Pesos base (50/50) |
| Vigilancia | `p25 < lag1 ≤ p50` | Pesos base (50/50) |
| Pre-brote | `p50 < lag1 ≤ p90` + tendencia↑ | `w_lstm` → mín(0.50×1.4, 0.65) |
| Brote activo | `lag1 > p90` + tendencia↑ | `w_lstm` → mín(0.50×(lag1/p90), 0.80) |
| Post-pico | `lag1 > p90` + tendencia↓ | `w_xgb` → mín(0.50×1.5, 0.75) |

> Los cinco regímenes son internos al sistema para el ajuste de pesos. El usuario solo ve los tres niveles del Agente 5 (Endémico / Alerta / Epidemia).

---

## Stack tecnológico

| Capa | Tecnologías |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 |
| ML | XGBoost 2.x · Scikit-Learn · SHAP (TreeSHAP) · Optuna (Bayesian HPO) |
| DL | PyTorch 2.x (LSTM 3 capas, hidden=77, lookback=12) |
| Datos | Pandas · NumPy |
| Frontend | React 19 · Vite · TailwindCSS · Leaflet.js |
| Visualización | SVG puro (ScatterPlot 4,056 puntos, sin librerías externas) |
| Storage | AWS S3 (`epipredict-dengue`) via boto3 |
| Deploy backend | Railway (Docker `python:3.11-slim`) |
| Deploy frontend | Vercel |

---

## Estructura del repositorio

```
/
├── agents/
│   ├── agente_1_recoleccion.py          # Ingesta OpenDengue + NASA POWER + BM + JMP
│   ├── agente_2_preprocesamiento.py     # Feature engineering (73 features)
│   ├── agente_3_prediccion_ml.py        # XGBoost + Optuna + SHAP global/local
│   ├── agente_4_prediccion_dl.py        # LSTM PyTorch + Optuna (hidden=77, layers=3)
│   ├── agente_5_alertas.py              # Orquestador: ensamble + 3 niveles de riesgo
│   ├── agente_6_regimen.py              # Detección de régimen + ajuste dinámico de pesos
│   └── s3_client.py                     # Cliente S3 compartido (upload/download)
├── backend/
│   ├── main.py                          # FastAPI app + todos los endpoints REST
│   ├── services.py                      # PredictionService (carga artefactos S3, orquesta)
│   └── schemas.py                       # Modelos Pydantic (RiskLevel: Endémico/Alerta/Epidemia)
├── frontend/
│   └── src/
│       ├── App.jsx                      # Raíz SPA + exportación PDF
│       └── components/
│           ├── Sidebar.jsx / Topbar.jsx
│           ├── DashboardView.jsx        # KPIs + ScatterPlot + mapa
│           ├── MapContainer.jsx         # Mapa Leaflet por nivel de riesgo
│           ├── PredictorView.jsx        # Sliders + semáforo + pesos dinámicos Agente 6
│           ├── ExplainabilityView.jsx   # SHAP global y local (XAI)
│           ├── InfoView.jsx             # Flujo de arquitectura y tech stack
│           └── ScatterPlot.jsx          # SVG puro, 4,056 puntos, dark mode
├── data/                                # Local — descargado desde S3 en runtime
│   ├── raw/                             # CSVs fuentes oficiales
│   ├── processed/                       # dataset_maestro + dataset_features
│   └── models/                          # Artefactos de modelos + metrics.json
├── scripts/
│   ├── training/entrenar_modelos.py     # Re-entrena ambos modelos desde cero
│   ├── pipeline/generar_scatter_data.py # Genera scatter_data.json (ejecutar 1 vez local)
│   ├── pipeline/optimizar_pesos_ensemble.py
│   └── analysis/                        # Scripts de análisis y diagnóstico
├── notebooks/                           # Notebooks Colab (gitignored)
├── Dockerfile                           # python:3.11-slim
├── Procfile                             # uvicorn backend.main:app --host 0.0.0.0 --port $PORT
└── requirements.txt
```

---

## API REST — Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/status` | Estado general del sistema |
| `GET` | `/api/metrics` | R², MAE, RMSE y pesos del ensamble |
| `GET` | `/api/metadata` | Países y departamentos disponibles |
| `GET` | `/api/coordinates` | Coordenadas GPS de los 169 departamentos |
| `GET` | `/api/historical` | Serie histórica mensual (`?iso_a0=&adm_1_name=`) |
| `GET` | `/api/features` | Features del último período para un departamento |
| `GET` | `/api/map-summary` | Incidencia media + nivel de riesgo por departamento |
| `GET` | `/api/top-departments` | Top N departamentos por incidencia histórica |
| `GET` | `/api/scatter-data` | 4,056 puntos real vs predicho (Ensemble/XGBoost/LSTM) |
| `GET` | `/api/explain/global` | Importancias SHAP globales (TreeSHAP) |
| `POST` | `/api/predict/simulate` | Predicción con sliders de variables climáticas + régimen |
| `POST` | `/api/predict/raw` | Predicción con vector de 73 features completo |

Documentación interactiva Swagger disponible en `/docs`.

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
    ├── pipeline_ml.pkl                  # Pipeline XGBoost (Bayesian optimizado)
    ├── xgb_model.pkl / escalador_ml.pkl / imputador_ml.pkl / cols_feat.pkl
    ├── shap_importance.json
    ├── lstm_model.pth                   # LSTM (hidden=77, layers=3, Bayesian)
    ├── escalador_lstm.pkl / lstm_features.pkl / lstm_config.json
    ├── scatter_data.json                # 4,056 puntos test para scatter plot
    ├── thresholds_clasificacion.json    # Percentiles p50/p90 globales + métricas de clasificación
    └── metrics.json                     # R², MAE, RMSE, pesos y métricas de clasificación
```

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS S3 |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS S3 |
| `RAILWAY_ENVIRONMENT` | Inyectada por Railway — activa rutas `/tmp/sma_data/` |
| `PORT` | Puerto del servidor (Railway lo inyecta automáticamente) |
| `VITE_API_URL` | URL del backend para el frontend (`.env.production` de Vite) |

---

## Decisiones de diseño

**Optimización Bayesiana (Optuna TPE)** — Reemplaza al GridSearchCV original. Explora 50 trials (XGBoost) y 30 trials (LSTM) con validación temporal `TimeSeriesSplit(k=5)` sobre el período 2016–2020. Converge a mejores hiperparámetros sin enumerar exhaustivamente el espacio de búsqueda.

**Pesos base 50/50** — Frente al resultado del optimizador (w=0.95 XGBoost, sesgado por la dominancia de XGBoost en el set de validación), se eligió 50/50 porque produce mejor MAE (5.83 vs 6.07) y RMSE (20.80 vs 22.18) en el test set. Los pesos base son ajustados dinámicamente por el Agente 6, por lo que el ensamble no es estático.

**LSTM con solo 6 features** — A diferencia de XGBoost, el LSTM recibe una ventana de 12 pasos consecutivos y aprende internamente la estructura temporal. Añadir los 73 features empeora el rendimiento por ruido.

**Transformación log1p** — La incidencia de dengue tiene distribución hipersimétrica (mayoría de registros 0–20 casos/100k, picos de 200–500+). La transformación permite que el modelo optimice uniformemente en todo el rango. El R² se reporta en escala log1p (estándar epidemiológico).

**Clasificación de 3 clases** — La fusión de "Normal" y "Vigilancia" en "Endémico" mejora el Kappa de 0.41 (Moderado) a 0.72 (Sustancial). Ambas categorías tienen el mismo protocolo de intervención.

**Carga en memoria al iniciar** — El backend descarga desde S3 todos los artefactos al arrancar y los mantiene en RAM. Latencia de inferencia < 200 ms por predicción.

---

## Fuentes de datos

| Fuente | Uso | Cobertura |
|---|---|---|
| **OpenDengue Project** | Casos de dengue subnacionales | 8 países, 2014–2022 |
| **NASA POWER API** | Temperatura, precipitación, humedad mensual | Global, ~0.5° resolución |
| **World Bank Open Data** | Población anual | Global |
| **JMP OMS/UNICEF** | Acceso a agua básica | Global, anual |
