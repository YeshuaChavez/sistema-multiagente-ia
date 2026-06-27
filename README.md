# EpiPredict — Sistema Multi-Agente SMA-ML/DL para Vigilancia Epidemiológica de Dengue

Sistema de alerta temprana para la predicción de epidemias de dengue a escala subnacional en América Latina. Implementa una arquitectura de **seis agentes inteligentes coordinados** que combinan Machine Learning (XGBoost + SHAP) y Deep Learning (LSTM PyTorch) en un ensamble híbrido con detección dinámica de régimen epidémico, desplegado como aplicación web completa en la nube.

**Proyecto Final — Universidad Nacional Mayor de San Marcos**  
Facultad de Ingeniería de Sistemas e Informática

---

## Demo

| Capa | URL |
|---|---|
| Frontend (Vercel) | https://proyecto-ia-weld.vercel.app |
| Backend API (Railway) | https://proyecto-ia-production.up.railway.app |
| Documentación interactiva (Swagger) | https://proyecto-ia-production.up.railway.app/docs |

---

## Métricas del sistema

| Modelo | R² (test 2021–2022) | MAE (casos/100k) | RMSE |
|---|---|---|---|
| XGBoost — Agente 3 | **91.23%** | 6.07 | 22.59 |
| LSTM PyTorch — Agente 4 | 86.94% | 6.23 | 20.63 |
| **Ensamble — Agente 5** | **91.06%** | **5.97** | **21.24** |

- **Conjunto de entrenamiento:** 12,168 observaciones mensuales (2014–2020)
- **Conjunto de prueba:** 4,056 observaciones (2021–2022), partición cronológica estricta
- **Conjunto de validación:** 1,014 observaciones (2020), usado exclusivamente para optimizar pesos del ensamble
- **Dataset features:** 16,224 registros × 73 variables predictoras
- **Pesos del ensamble:** `w_xgb = 0.90`, `w_lstm = 0.10` (optimizados sobre val 2020 con `scipy.optimize`, sin data leakage sobre test)
- **Cobertura:** 8 países, 169 unidades subnacionales
- Las métricas R² se reportan en escala `log1p`, estándar para distribuciones epidemiológicas asimétricas

### Clasificación de riesgo epidémico

| Métrica | 4 clases (anterior) | 3 clases (actual) |
|---|---|---|
| Accuracy | 57.2% | **84.8%** |
| Cohen's Kappa | 0.411 (Moderado) | **0.708 (Sustancial)** |
| Epidemia — Precision | 86.5% | **91.0%** |
| Epidemia — Recall | 58.0% | **83.3%** |
| Epidemia — F1 | 69.5% | **87.0%** |

El sistema de 3 clases (**Endémico / Alerta / Epidemia**) fusiona las categorías Normal y Vigilancia en "Endémico", validado como equivalente epidemiológicamente (ambas representan condiciones no urgentes). La mejora en Kappa de 0.411 a 0.708 es sustancial y supera el umbral de 0.60 considerado aceptable para sistemas de alerta en salud pública.

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
┌──────────────────────────────────────────────────────────┐
│               AWS S3  (epipredict-dengue)                │
│   datos_crudos/  ·  datos_procesados/  ·  modelos/       │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  Agente 1 — Recolección
  (ingesta automática desde fuentes oficiales → datos_crudos/)
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
  R²=91.23%  R²=86.94%
     └──┬──┘
        ▼
  Agente 5 — Orquestador
  (ensamble ponderado R²=91.06% + clasificación Endémico/Alerta/Epidemia por percentiles)
        │
        ▼ (consulta régimen)
  Agente 6 — Régimen Epidémico
  (ajuste dinámico de pesos según fase: Normal / Vigilancia / Pre-brote / Brote / Post-pico)
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

Almacena los artefactos crudos en S3 bajo `datos_crudos/`.

---

### Agente 2 — Preprocesamiento y Feature Engineering

Calcula la tasa de incidencia mensual normalizada (`casos / población × 100,000`) y construye las **73 variables predictoras** a partir de 8 grupos:

| Grupo | Variables | Cantidad |
|---|---|---|
| Base climática y demográfica | `tmax`, `tmin`, `precipitacion`, `humedad`, `poblacion`, `densidad_poblacion` | 6 |
| Lags climáticos | `tmax/tmin/precipitacion/humedad lag1–lag6` | 24 |
| Lags de incidencia | `incidencia_lag1` a `incidencia_lag12` (en escala `log1p`) | 12 |
| Rolling means de incidencia | `incidencia_roll3`, `roll6`, `roll12` (en escala `log1p`) | 3 |
| Vecinos espaciales | `incidencia_vecinos_lag1–lag6` (promedio de los 3 departamentos más cercanos por GPS, mismo país) | 6 |
| Estacionalidad cíclica | `mes_sin = sin(2π·mes/12)`, `mes_cos = cos(2π·mes/12)` | 2 |
| Indicadores epidemiológicos | `indicador_covid`, `indicador_nino`, `indicador_nina` | 3 |
| Features derivadas | `amplitud_termica`, `temperatura_media`, `precipitacion_anomalia`, `aceleracion_incidencia`, `cambio_interanual`, `tendencia_1m`, `tendencia_3m`, `fase_ascendente`, `indicador_brote` | 9 |
| Dummies de país | `pais_ARG`, `pais_BOL`, `pais_BRA`, `pais_COL`, `pais_ECU`, `pais_MEX`, `pais_PAN`, `pais_PER` | 8 |
| **Total** | | **73** |

Produce dos artefactos en S3:

- `datos_procesados/dataset_maestro_mensual_latam.csv` — 18,252 filas × 14 columnas (base para histórico e inferencia)
- `datos_procesados/dataset_features_latam.csv` — 16,224 filas × 81 columnas (73 features para entrenamiento)

---

### Agente 3 — Predicción ML (XGBoost + SHAP)

Pipeline completo con transformación logarítmica del target para manejar la distribución asimétrica de la incidencia:

```
SimpleImputer(median) → StandardScaler → XGBRegressor
target: log1p(incidencia_dengue) → output: expm1(prediccion)
```

**Hiperparámetros (GridSearchCV + TimeSeriesSplit):**

```
n_estimators  = 800
learning_rate = 0.01
max_depth     = 4
random_state  = 42
```

Calcula importancias **SHAP globales** (TreeSHAP, promedio sobre el test set completo) y **SHAP locales** por predicción individual. Serializa todos los artefactos a S3 (`modelos/`).

**Resultado en test 2021–2022:** R² = 91.23% | MAE = 6.07 casos/100k | RMSE = 22.59

---

### Agente 4 — Predicción DL (LSTM PyTorch)

Red LSTM de dos capas apiladas que aprende dependencias temporales de largo alcance directamente desde la secuencia. Usa **solo 6 features** (sin lags explícitos — la memoria interna del LSTM los reemplaza):

```
Features: tmax_promedio · tmin_promedio · precipitacion ·
          humedad_promedio · agua_basica · incidencia_dengue
```

**Arquitectura:**

```
Input:    lookback=12 meses × 6 features
LSTM:     hidden_dim=256, num_layers=2, dropout=0.1
Output:   Linear(256 → 1) → expm1 → incidencia predicha
```

**Entrenamiento:** Adam (`lr=0.003`), Early Stopping, ReduceLROnPlateau, `torch.manual_seed(42)`, CPU-only (reproducible).

**Resultado en test 2021–2022:** R² = 86.94% | MAE = 6.23 casos/100k | RMSE = 20.63

---

### Agente 5 — Orquestador de Consenso (Ensamble + Alertas)

Combina las predicciones de los Agentes 3 y 4 con **pesos optimizados sobre el conjunto de validación 2020** (sin data leakage — los pesos nunca se estiman sobre el test set 2021–2022):

```python
w_xgb  = 0.90   # optimizado con scipy.optimize.minimize_scalar sobre val 2020
w_lstm = 0.10

pred_ensemble = w_xgb × pred_xgb + w_lstm × pred_lstm
```

Los pesos base son ajustados dinámicamente por el **Agente 6** según el régimen epidémico detectado. En brotes activos, `w_lstm` puede elevarse hasta 0.80 para capturar el momentum temporal que XGBoost subestima.

Clasifica cada departamento en **tres niveles de riesgo** usando **percentiles históricos locales calibrados por departamento** (train 2014–2020):

| Nivel | Criterio | Color | Acción |
|---|---|---|---|
| **Endémico** | predicción ≤ p50 histórico del departamento | Verde | Monitoreo rutinario |
| **Alerta** | p50 < predicción ≤ p90 | Naranja | Fumigaciones selectivas, cerco epidemiológico |
| **Epidemia** | predicción > p90 | Rojo | Emergencia de salud pública, intervención inmediata |

El p90 usa un floor global: `p90_efectivo = max(p90_local, p90_global)` para evitar que departamentos con histórico de incidencia muy baja generen falsas alarmas de epidemia con valores absolutamente irrelevantes.

**Resultado en test 2021–2022:** R² = 91.06% | MAE = 5.97 casos/100k | RMSE = 21.24  
**Clasificación (3 clases):** Accuracy = 84.8% | Cohen's Kappa = 0.708

> El valor diferencial del ensamble sobre XGBoost solo no radica en el R² global estático sino en el comportamiento dinámico: en fase de brote activo, el Agente 6 eleva el peso del LSTM hasta 0.80 porque el LSTM captura mejor la aceleración temporal de la incidencia en fases de crecimiento explosivo, mientras XGBoost —basado en árboles— no puede extrapolar más allá del rango histórico máximo de cada región.

---

### Agente 6 — Régimen Epidémico (Ajuste Dinámico de Pesos)

Detecta el régimen epidemiológico actual de cada departamento comparando `incidencia_lag1` contra percentiles históricos locales (`p25`, `p50`, `p90`) y la tendencia de la serie (`lag1_log - lag2_log`).

**Floor global:** `p90_efectivo = max(p90_local, p90_global)` — evita falsas alarmas en departamentos de baja endemia histórica.

| Régimen | Condición | Ajuste de pesos |
|---|---|---|
| Normal | `lag1 ≤ p25` | Pesos base (w_xgb=0.90) |
| Vigilancia | `p25 < lag1 ≤ p50` | Pesos base (w_xgb=0.90) |
| Pre-brote | `p50 < lag1 ≤ p90` + tendencia↑ | `w_lstm` → mín(base×1.4, 0.65) |
| Brote activo | `lag1 > p90` + tendencia↑ | `w_lstm` → mín(base×(lag1/p90), 0.80) |
| Post-pico | `lag1 > p90` + tendencia↓ | `w_xgb` → mín(base×1.5, 0.75) |

> Nota: los cinco regímenes del Agente 6 son internos al sistema para el ajuste de pesos. Los niveles de riesgo visibles al usuario son los tres del Agente 5 (Endémico / Alerta / Epidemia).

El Agente 5 carga el Agente 6 en tiempo de inferencia mediante `importlib.util.spec_from_file_location` para evitar conflictos de `sys.path` en el entorno Railway.

---

## Stack tecnológico

| Capa | Tecnologías |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 |
| ML | XGBoost 2.x · Scikit-Learn · SHAP (TreeSHAP) · SciPy (optimización de pesos) |
| DL | PyTorch 2.x (LSTM 2 capas, hidden=256, lookback=12) |
| Datos | Pandas · NumPy |
| Frontend | React 19 · Vite · TailwindCSS · Leaflet.js |
| Visualización | SVG puro (ScatterPlot 4,056 puntos, sin librerías externas) |
| Storage | AWS S3 (`epipredict-dengue`) via boto3 |
| Deploy backend | Railway (Docker `python:3.11-slim`) |
| Deploy frontend | Vercel |

---

## Estructura del repositorio

```
proyecto-ia/
├── agentes/
│   ├── agente_1_recoleccion.py          # Ingesta OpenDengue + NASA POWER + BM + JMP
│   ├── agente_2_preprocesamiento.py     # Feature engineering (73 features)
│   ├── agente_3_prediccion_ml.py        # XGBoost + SHAP global/local
│   ├── agente_4_prediccion_dl.py        # LSTM PyTorch (hidden=256, lookback=12)
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
│       ├── main.jsx                     # Entry point React
│       └── components/
│           ├── Sidebar.jsx
│           ├── Topbar.jsx               # Modo oscuro
│           ├── DashboardView.jsx        # KPIs globales + ScatterPlot + mapa
│           ├── MapContainer.jsx         # Mapa Leaflet geoespacial por riesgo
│           ├── PredictorView.jsx        # Sliders + resultado + card régimen Agente 6
│           ├── ExplainabilityView.jsx   # SHAP global y local (XAI)
│           └── InfoView.jsx             # Documentación técnica del sistema
├── Base de Datos/
│   ├── datos_crudos/                    # CSVs fuentes oficiales (no en git)
│   ├── datos_procesados/                # dataset_maestro + dataset_features (no en git)
│   └── modelos/
│       └── metrics.json                 # Métricas + pesos del ensamble (en git)
├── generar_scatter_data.py              # Genera scatter_data.json y actualiza metrics.json en S3
├── optimizar_pesos_ensemble.py          # Optimiza w_xgb/w_lstm con scipy sobre val 2020
├── Dockerfile                           # python:3.11-slim, CMD uvicorn $PORT
├── requirements.txt
└── .env                                 # Credenciales locales (no commiteado)
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
│   ├── Temporal_extract_V1_3.csv          # Casos dengue (OpenDengue)
│   ├── clima_nasa_crudo.csv               # Variables climáticas NASA POWER
│   ├── agua_jmp_crudo.csv                 # Acceso a agua JMP OMS/UNICEF
│   ├── departamentos_coordenadas.csv      # Coordenadas GPS (169 departamentos)
│   └── poblacion/                         # Censos por país
├── datos_procesados/
│   ├── dataset_maestro_mensual_latam.csv  # 18,252 filas × 14 cols (inferencia)
│   └── dataset_features_latam.csv         # 16,224 filas × 81 cols (entrenamiento)
└── modelos/
    ├── pipeline_ml.pkl                    # Pipeline completo XGBoost
    ├── xgb_model.pkl                      # Modelo XGBoost serializado
    ├── imputador_ml.pkl                   # SimpleImputer (mediana)
    ├── escalador_ml.pkl                   # StandardScaler para XGBoost
    ├── cols_feat.pkl                      # Lista de 73 features en orden
    ├── shap_importance.json               # Importancias SHAP globales
    ├── lstm_model.pth                     # Pesos del modelo LSTM
    ├── escalador_lstm.pkl                 # StandardScaler para secuencias LSTM
    ├── lstm_features.pkl                  # Lista de 6 features LSTM
    ├── lstm_config.json                   # Configuración de arquitectura LSTM
    ├── scatter_data.json                  # 4,056 puntos test (3 modelos) para scatter plot
    └── metrics.json                       # R², MAE, RMSE y pesos w_xgb/w_lstm del ensamble
```

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

## Fuentes de datos

| Fuente | Uso | Cobertura |
|---|---|---|
| **OpenDengue Project** | Casos de dengue a escala subnacional | 8 países, 2014–2022 |
| **NASA POWER API** (NASA Langley) | Temperatura máx/mín, precipitación, humedad mensual | Global, resolución ~0.5° |
| **World Bank Open Data** | Estimaciones de población anual | Global, anual |
| **JMP OMS/UNICEF** | Indicador oficial de acceso a agua básica | Global, anual |

---

## Reproducibilidad

| Componente | Semilla / Estrategia |
|---|---|
| XGBoost | `random_state=42` |
| LSTM PyTorch | `torch.manual_seed(42)` + `numpy.random.seed(42)`, CPU-only |
| Split temporal | Train ≤ 2020 / Val 2020 / Test 2021–2022, partición estricta sin data leakage |
| Pesos ensamble | Optimizados con `scipy.optimize.minimize_scalar` sobre val 2020 exclusivamente |

Reentrenar desde cero desde los mismos datos produce resultados idénticos.

---

## Decisiones de diseño

**Pesos del ensamble optimizados sobre val 2020** — Se busca `w_xgb ∈ [0.20, 0.90]` con `minimize_scalar` sobre el conjunto de validación (2020), minimizando el RMSE del ensamble en escala `log1p`. El resultado óptimo fue `w_xgb=0.90`, que refleja la superioridad de XGBoost en la mayor parte del espacio de incidencia. El LSTM complementa en fase dinámica de brote (ver Agente 6). Los pesos nunca se estiman sobre el test set 2021–2022.

**Por qué el LSTM usa solo 6 features** — A diferencia de XGBoost, que necesita lags explícitos para capturar historia, el LSTM recibe una ventana de 12 pasos consecutivos y aprende internamente la estructura temporal. Añadir los 73 features empeora el rendimiento por ruido y complejidad innecesaria.

**Transformación log1p** — La incidencia de dengue tiene distribución hipersimétrica (miles de registros con 0–20 casos/100k y picos puntuales de 200–500+). Sin la transformación logarítmica, los árboles y redes optimizan los extremos y el R² colapsa. El target siempre es `log1p(incidencia)` al entrenar; la salida al usuario es `expm1(prediccion)`. El R² reportado se calcula también en escala `log1p` (estándar epidemiológico).

**Clasificación de 3 clases en lugar de 4** — La fusión de las categorías "Normal" y "Vigilancia" en "Endémico" está respaldada epidemiológicamente (ambas representan condiciones de baja urgencia sin diferencia en el protocolo de intervención) y estadísticamente mejora el Cohen's Kappa de 0.411 (Moderado) a 0.708 (Sustancial). La clasificación es un output secundario; la métrica primaria del sistema es R²=91.06%.

**Floor global p90 en el Agente 6** — `p90_efectivo = max(p90_local, p90_global)` evita que departamentos con histórico de incidencia baja (p90 local = 5 casos/100k) generen alertas de "brote activo" con valores absolutamente irrelevantes en términos de salud pública.

**Inferencia de vecinos en tiempo real** — Durante la predicción online, `incidencia_vecinos_lag1–6` se calcula buscando los 3 departamentos más cercanos en el mapa GPS y promediando su incidencia histórica real. No se usan aproximaciones ni fallbacks salvo ausencia total de datos históricos del vecino.

**Carga en memoria al iniciar** — El backend descarga desde S3 todos los artefactos (modelos, escaladores, dataset maestro, mapa de vecinos) al arrancar y los mantiene en RAM. Latencia de inferencia: < 200 ms por predicción.
