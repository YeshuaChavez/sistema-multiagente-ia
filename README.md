# Sistema Multi-Agente para la Prediccion de Incidencia de Dengue en Latinoamerica (SMA-ML/DL)

Este proyecto implementa un Sistema Multi-Agente Cooperativo para la prediccion y analisis explicable de la tasa de incidencia de dengue a nivel subnacional (departamental) mensual en Latinoamerica, cubriendo el periodo historico de 2014 a 2022. Desarrollado por Yeshua Chavez.

## Arquitectura del Sistema Multi-Agente (SMA-ML/DL)

La solucion se compone de 5 agentes informaticos especializados que cooperan de manera secuencial y asincrona:

1. **Agente 1: Recoleccion (agentes/agente_1_recoleccion.py)**:
   * Encargado de la ingesta automatizada y dirigida de multiples fuentes de datos locales y online.
   * Carga los censos poblacionales de 9 paises (Argentina, Bolivia, Brasil, Colombia, Ecuador, Mexico, Nicaragua, Panama y Peru) y unifica sus esquemas temporales.
   * Resuelve coordenadas geograficas de los departamentos usando Nominatim con cache local.
   * Realiza consultas dirigidas a la API mensual oficial de NASA POWER para variables climaticas y a la API del Banco Mundial para el indicador de acceso a agua potable (JMP).

2. **Agente 2: Preprocesamiento (agentes/agente_2_preprocesamiento.py)**:
   * Une los datasets de casos (OpenDengue), clima (NASA POWER), agua basica (JMP) y poblacion.
   * Normaliza los casos a tasa de incidencia por cada 100,000 habitantes.
   * Estructura rezagos simetricos de 1 a 3 meses (lags 1, 2 y 3) para temperatura maxima, temperatura minima, precipitacion, humedad relativa y la propia incidencia autoregresiva.
   * Genera el archivo consolidado final `Base de Datos/dataset_maestro_mensual_latam.csv`.

3. **Agente 3: Prediccion Machine Learning (agentes/agente_3_prediccion_ml.py)**:
   * Entrena el regresor XGBoost (Extreme Gradient Boosting) bajo una particion cronologica estricta:
     * Entrenamiento: Años 2014-2020 (15,147 registros).
     * Prueba: Años 2021-2022 (4,488 registros).
   * Valida internamente con validacion cruzada K-Fold (K=5) sobre el bloque de entrenamiento.
   * Genera la capa de explicabilidad algorítmica global (XAI) mediante el calculo de valores SHAP (Shapley Additive exPlanations) usando TreeSHAP.

4. **Agente 4: Prediccion Deep Learning (agentes/agente_4_prediccion_dl.py)**:
   * Entrena una red neuronal recurrente LSTM (Long Short-Term Memory) implementada en PyTorch (ejecucion en CPU).
   * Modela secuencias historicas de 3 meses ($t-3, t-2, t-1$) integrando covariables estaticas del mes de proyeccion ($t$) para pronosticos no lineales.

5. **Agente 5: Alertas y GUI (agentes/agente_5_alertas.py)**:
   * Consolida los pronosticos individuales en un modelo Ensemble promedio (XGBoost + LSTM).
   * Clasifica la tasa de incidencia proyectada en 4 niveles de riesgo (Bajo/Normal, Vigilancia, Alerta, Epidemia) basandose en percentiles historicos.
   * Aloja la interfaz grafica interactiva interactores de sliders en vivo, renders de dashboard y graficos comparativos de metricas en Tkinter.

## Requisitos del Sistema

* Python 3.8 o superior
* PyTorch (CPU)
* XGBoost
* SHAP
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Requests

Para instalar las dependencias necesarias:

```bash
pip install torch xgboost shap scikit-learn pandas numpy matplotlib seaborn requests
```

## Estructura del Workspace

```text
├── agentes/
│   ├── __init__.py
│   ├── agente_1_recoleccion.py
│   ├── agente_2_preprocesamiento.py
│   ├── agente_3_prediccion_ml.py
│   ├── agente_4_prediccion_dl.py
│   └── agente_5_alertas.py
├── Base de Datos/
│   ├── Temporal_extract_V1_3.csv       (Casos crudos de OpenDengue)
│   ├── departamentos_coordenadas.csv    (Cache geografica)
│   ├── clima_nasa_crudo.csv            (Clima subnacional mensual de NASA POWER)
│   ├── agua_jmp_crudo.csv              (Agua potable subnacional de JMP/WB)
│   ├── poblacion_*.csv                 (Censos de poblacion gubernamentales por pais)
│   └── dataset_maestro_mensual_latam.csv (Generado por el Agente 2)
├── Scripts/                            (Scripts auxiliares e historicos)
├── main.py                             (Lanzador principal)
└── README.md                           (Este archivo)
```

## Instrucciones de Ejecucion

Para ejecutar el lanzador principal del sistema y desplegar la GUI interactiva:

1. Abra una terminal en el directorio del proyecto:
   ```bash
   cd "c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
   ```

2. Ejecute el comando:
   ```bash
   python main.py
   ```

El lanzador inicializara secuencialmente el flujo de los agentes, entrenara los modelos predictivos en segundo plano (durante aproximadamente 15 segundos) y abrira la interfaz en su escritorio.
