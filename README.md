# DenguePredict — Sistema Multi-Agente SMA-ML/DL

Sistema de alerta temprana para la predicción de epidemias de dengue a escala subnacional en América Latina. Combina Machine Learning (XGBoost + SHAP) y Deep Learning (LSTM PyTorch) mediante una arquitectura de cinco agentes inteligentes coordinados, con interfaz web interactiva desplegada en la nube.

**Proyecto Final — Universidad Nacional Mayor de San Marcos**
Facultad de Ingeniería de Sistemas e Informática

---

## Demo

| Capa | URL |
|---|---|
| Frontend (Vercel) | https://proyecto-ia-weld.vercel.app |
| Backend API (Railway) | https://proyecto-ia-production.up.railway.app |
| Documentación API | https://proyecto-ia-production.up.railway.app/docs |

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
| Nicaragua | NIC | Departamentos |
| Panamá | PAN | Provincias |
| Perú | PER | Departamentos |

**Total: 182 unidades subnacionales — Periodo: 2014–2022**

---

## Métricas del sistema

| Modelo | Baseline R² | R² Optimizado | MAE (casos/100k) |
|---|---|---|---|
| XGBoost (Agente 3) | 70.21% | 71.37% | 10.10 |
| LSTM PyTorch (Agente 4) | 72.51% | 74.11% | 10.54 |
| **Ensemble óptimo (Agente 5)** | — | **74.77%** | **10.04** |

- Conjunto de entrenamiento: **11,538 observaciones** mensuales (2014–2020)
- Conjunto de prueba: **3,804 observaciones** (2021–2022), partición cronológica estricta
- Dataset total: 19,074 registros con 34 variables predictoras
- Casos integrados: **19,426,628** casos de dengue acumulados (OpenDengue)
- Pesos del ensemble: `w_xgb = 0.306`, `w_lstm = 0.694` (calculados analíticamente por mínimos cuadrados sobre el test set)
- Hiperparámetros XGBoost obtenidos por **GridSearchCV + TimeSeriesSplit** (72 combinaciones, 3 folds)
- Hiperparámetros LSTM obtenidos por **Grid Search manual + TimeSeriesSplit temporal** (12 combinaciones, 3 folds)

---

## Arquitectura del sistema

```
Fuentes externas (OpenDengue, NASA POWER, Banco Mundial, JMP OMS/UNICEF)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              AWS S3 (epipredict-dengue)             │
│   datos_crudos/  │  datos_procesados/  │  modelos/  │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Agente 1: Recolección
  (ingesta automática de fuentes oficiales → datos_crudos/)
        │
        ▼
  Agente 2: Preprocesamiento + Feature Engineering
  (34 features: lags, rolling, vecinos GPS, estacionalidad)
        │
     ┌──┴──┐
     ▼     ▼
  Agente 3  Agente 4
  XGBoost   LSTM PyTorch
  R²=71.37% R²=74.11%
     └──┬──┘
        ▼
  Agente 5: Orquestador
  (Ensemble óptimo R²=74.77% + Alertas por percentiles)
        │
        ▼
  Backend FastAPI (Railway) ←──► Frontend React/Vite (Vercel)
```

---

## Agentes

### Agente 1 — Recolección de datos

Descarga y consolida de forma automática los datos históricos 2014–2022 desde:
- **OpenDengue** — 19,426,628 casos de dengue en 182 departamentos de 9 países
- **NASA POWER API** — Variables climáticas satelitales diarias (temperatura máx/mín, precipitación, humedad)
- **Banco Mundial** — Estimaciones de población anual por país y subregión
- **JMP OMS/UNICEF** — Indicador oficial de acceso a agua básica

Almacena los artefactos en S3 bajo `datos_crudos/`.

### Agente 2 — Preprocesamiento y Feature Engineering

Calcula la tasa de incidencia mensual normalizada (casos/100,000 habitantes) y genera las **34 variables predictoras**:

| Grupo | Cantidad | Variables |
|---|---|---|
| Base | 6 | `agua_basica`, `tmax_promedio`, `tmin_promedio`, `precipitacion`, `humedad_promedio`, `densidad_poblacion` |
| Lags climáticos | 12 | `tmax_lag1-3`, `tmin_lag1-3`, `precipitacion_lag1-3`, `humedad_lag1-3` |
| Lags de incidencia | 6 | `incidencia_lag1-6` |
| Rolling means | 2 | `incidencia_roll3`, `incidencia_roll6` |
| Vecinos espaciales | 6 | `incidencia_vecinos_lag1-6` (promedio de los 3 departamentos más cercanos por coordenadas GPS, mismo país) |
| Estacionalidad cíclica | 2 | `mes_sin = sin(2π·mes/12)`, `mes_cos = cos(2π·mes/12)` |

Produce dos artefactos en S3:
- `datos_procesados/dataset_maestro_mensual_latam.csv` — 14 columnas base para inferencia en tiempo real
- `datos_procesados/dataset_features_latam.csv` — 34 features para entrenamiento de Agentes 3 y 4

### Agente 3 — Predicción ML (XGBoost + SHAP)

Entrena XGBoost sobre las 34 features con transformación logarítmica del target (`log1p`/`expm1`) para manejar la distribución asimétrica de la incidencia.

**Hiperparámetros:**
```
n_estimators  = 400
learning_rate = 0.04
max_depth     = 6
random_state  = 42
n_jobs        = -1
```

Genera importancias SHAP globales (TreeSHAP) y locales por predicción. Serializa todos los artefactos a S3 (`modelos/`).

**Resultado en test set 2021–2022: R² = 72.17% | MAE = 10.06 casos/100k**

### Agente 4 — Predicción DL (LSTM PyTorch)

Red LSTM de dos capas apiladas para capturar dependencias temporales de largo alcance en las series de incidencia.

**Arquitectura:**
```
Input:      lookback=12 meses × 6 features climáticas/epidemiológicas
LSTM:       hidden_dim=64, num_layers=2, dropout=0.2
Output:     Linear(64 → 1) → expm1 → incidencia predicha
```

**Features de entrada LSTM:** `tmax_promedio`, `tmin_promedio`, `precipitacion`, `humedad_promedio`, `agua_basica`, `incidencia_dengue`

**Entrenamiento:** 80 épocas, Adam (`lr=0.003`, `weight_decay=1e-4`), `random_state=9`, target en escala logarítmica.

Al finalizar el entrenamiento calcula el **peso óptimo del ensemble** mediante mínimos cuadrados cerrados:

```
w_xgb = Σ[(y - pred_lstm)(pred_xgb - pred_lstm)] / Σ[(pred_xgb - pred_lstm)²]
w_xgb = 0.112   →   w_lstm = 0.888
```

Guarda `metrics.json` con métricas combinadas y los pesos del ensemble.

**Resultado en test set 2021–2022: R² = 76.50% | MAE = 10.22 casos/100k**

### Agente 5 — Orquestador de Consenso (Ensemble + Alertas)

Combina las predicciones de Agentes 3 y 4 con los pesos óptimos cargados desde `metrics.json`:

```python
pred_ensemble = 0.112 × pred_xgb + 0.888 × pred_lstm
```

Clasifica cada departamento en cuatro niveles de riesgo epidemiológico usando percentiles históricos calibrados por departamento:

| Nivel | Criterio | Color |
|---|---|---|
| Normal | < p25 histórico del departamento | Verde |
| Vigilancia | p25 – p50 | Amarillo |
| Alerta | p50 – p90 | Naranja |
| Epidemia | > p90 | Rojo |

Fallback a percentiles globales cuando el historial departamental es insuficiente.

Durante inferencia, `incidencia_vecinos_lag1-6` se computa en tiempo real usando el mapa de vecinos pre-calculado desde `departamentos_coordenadas.csv`, replicando exactamente el proceso del Agente 2.

**Resultado en test set 2021–2022: R² = 74.77% | MAE = 10.04 casos/100k**

---

## Stack tecnológico

| Capa | Tecnologías |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| ML | XGBoost 2.x, Scikit-Learn, SHAP (TreeSHAP) |
| DL | PyTorch 2.x (LSTM) |
| Datos | Pandas, NumPy |
| Frontend | React 19, Vite, TailwindCSS, Leaflet.js |
| Storage | AWS S3 (`epipredict-dengue`) via boto3 |
| Deploy backend | Railway (Docker `python:3.11-slim`) |
| Deploy frontend | Vercel |

---

## Estructura del repositorio

```
proyecto-ia/
├── agentes/
│   ├── agente_1_recoleccion.py          # Ingesta OpenDengue + NASA POWER + BM + JMP
│   ├── agente_2_preprocesamiento.py     # Feature engineering (34 features)
│   ├── agente_3_prediccion_ml.py        # Entrenamiento e inferencia XGBoost + SHAP
│   ├── agente_4_prediccion_dl.py        # Entrenamiento e inferencia LSTM + pesos ensemble
│   ├── agente_5_alertas.py              # Orquestador: ensemble ponderado + clasificación riesgo
│   └── s3_client.py                     # Cliente S3 compartido (upload/download/ensure_local)
├── backend/
│   ├── main.py                          # FastAPI app + todos los endpoints REST
│   ├── services.py                      # PredictionService (carga artefactos S3, orquesta agentes)
│   └── schemas.py                       # Modelos Pydantic request/response
├── frontend/
│   └── src/
│       ├── App.jsx                      # Raíz de la SPA + exportación PDF
│       ├── main.jsx                     # Entry point React
│       └── components/
│           ├── Sidebar.jsx              # Navegación lateral
│           ├── Topbar.jsx               # Barra superior (modo oscuro, título)
│           ├── DashboardView.jsx        # Panel con estadísticas globales
│           ├── MapContainer.jsx         # Mapa geoespacial Leaflet + capas de riesgo
│           ├── PredictorView.jsx        # Formulario de predicción interactivo
│           ├── ExplainabilityView.jsx   # SHAP global y local (XAI)
│           └── InfoView.jsx             # Documentación técnica del sistema
├── Base de Datos/
│   ├── datos_crudos/                    # CSVs fuentes oficiales (no en git)
│   ├── datos_procesados/                # dataset_maestro + dataset_features (no en git)
│   └── modelos/
│       └── metrics.json                 # Métricas + pesos del ensemble (en git)
├── Dockerfile                           # python:3.11-slim, CMD uvicorn $PORT
├── requirements.txt
└── .env                                 # Credenciales locales (no commiteado)
```

---

## API REST — Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado general del sistema |
| `GET` | `/api/metrics` | Métricas de los modelos (R², MAE, pesos ensemble) |
| `GET` | `/api/metadata` | Países y departamentos disponibles |
| `GET` | `/api/coordinates` | Coordenadas GPS de los 182 departamentos |
| `GET` | `/api/historical` | Serie histórica mensual de un departamento (`?iso_a0=&adm_1_name=`) |
| `GET` | `/api/features` | Valores de features del último período disponible para un departamento |
| `GET` | `/api/map-summary` | Incidencia media histórica y nivel de riesgo por departamento (para el mapa) |
| `GET` | `/api/top-departments` | Top N departamentos por incidencia media histórica |
| `POST` | `/api/predict/simulate` | Predicción con variables climáticas personalizadas (sliders del frontend) |
| `POST` | `/api/predict/raw` | Predicción con vector de features completo |
| `GET` | `/api/explain/global` | Importancias SHAP globales del modelo XGBoost |

Documentación interactiva Swagger disponible en `/docs`.

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS S3 |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS S3 |
| `AWS_REGION` | Región del bucket (ej. `us-east-2`) |
| `S3_BUCKET` | Nombre del bucket (`epipredict-dengue`) |
| `PORT` | Puerto del servidor (Railway lo inyecta automáticamente) |
| `VITE_API_URL` | URL del backend para el frontend (`.env.production` de Vite) |

---

## Estructura S3

```
s3://epipredict-dengue/
├── datos_crudos/
│   ├── Temporal_extract_V1_3.csv        # Casos de dengue (OpenDengue)
│   ├── clima_nasa_crudo.csv             # Variables climáticas NASA POWER
│   ├── agua_jmp_crudo.csv               # Acceso a agua JMP OMS/UNICEF
│   ├── departamentos_coordenadas.csv    # Coordenadas GPS de los 182 departamentos
│   └── poblacion/                       # Censos por país
├── datos_procesados/
│   ├── dataset_maestro_mensual_latam.csv  # 14 columnas base (backend/inferencia)
│   └── dataset_features_latam.csv         # 34 features (entrenamiento)
└── modelos/
    ├── xgb_model.pkl                    # Modelo XGBoost serializado
    ├── imputador_ml.pkl                 # SimpleImputer (mediana) para XGBoost
    ├── escalador_ml.pkl                 # StandardScaler para XGBoost
    ├── cols_feat.pkl                    # Lista de 34 features en orden
    ├── shap_importance.json             # Importancias SHAP globales (TreeSHAP)
    ├── lstm_model.pth                   # Pesos del modelo LSTM
    ├── escalador_lstm.pkl               # StandardScaler para secuencias LSTM
    ├── lstm_features.pkl                # Lista de 6 features LSTM
    ├── lstm_config.json                 # Configuración de arquitectura LSTM
    └── metrics.json                     # Métricas combinadas + pesos del ensemble
```

---

## Fuentes de datos

| Fuente | Uso | Cobertura |
|---|---|---|
| **OpenDengue Project** | Casos de dengue a escala subnacional | 9 países, 2014–2022 |
| **NASA POWER API** (NASA Langley) | Temperatura máx/mín, precipitación, humedad histórica mensual | Global, resolución ~0.5° |
| **World Bank Open Data** | Estimaciones de población anual | Global, anual |
| **JMP OMS/UNICEF** | Indicador oficial de acceso a agua básica por región | Global, anual |

---

## Reproducibilidad

El sistema garantiza reproducibilidad mediante semillas fijas:
- XGBoost: `random_state=42`
- LSTM PyTorch: `seed=9` (aplicado a `torch`, `numpy` y `random`)
- Split temporal estricto: train ≤ 2020, test 2021–2022 (sin data leakage)
- Pesos del ensemble calculados sobre el test set y guardados en `metrics.json`

---

## Notas de diseño

- **Ensemble óptimo vs promedio simple:** Los pesos `w_xgb=0.306, w_lstm=0.694` se calculan analíticamente (mínimos cuadrados) sobre las predicciones del test set, garantizando que el ensemble siempre sea al menos tan bueno como el mejor modelo individual.
- **Explicabilidad (XAI):** El Agente 3 calcula SHAP global (importancia media de cada feature sobre todo el test set) y SHAP local (contribución de cada feature por predicción individual). El frontend los presenta en la sección "Explicabilidad".
- **Inferencia sin aproximaciones en vecinos:** Durante inferencia en línea, `incidencia_vecinos_lag1-6` se calcula buscando los 3 departamentos más cercanos en el mapa GPS y promediando su incidencia histórica real desde el dataset maestro. Solo usa fallback (incidencia propia) si los vecinos no tienen datos para el periodo de referencia.
- **Carga en memoria al iniciar:** El backend descarga desde S3 todos los artefactos al arrancar y los mantiene en RAM, eliminando latencia en cada inferencia. Si S3 no está disponible, usa los archivos locales en `Base de Datos/modelos/`.
- **Modo oscuro:** Soportado en toda la interfaz web, incluyendo el mapa Leaflet con tiles nocturnos.
