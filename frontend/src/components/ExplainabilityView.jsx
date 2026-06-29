import React, { useEffect, useState, useCallback } from "react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Diccionario de explicaciones biológicas por variable SHAP ───
// icon: nombre de Material Symbol
const SHAP_BIO = {
  // Incidencia autorregresiva
  incidencia_lag1:  { icon: "biotech",          title: "Incidencia hace 1 mes",   bio: "El predictor más potente. Si hay muchos casos este mes, el virus ya circula activamente: reservorio humano infectado, mosquitos vectores picando, condiciones propicias. Captura el momentum epidémico." },
  incidencia_lag2:  { icon: "biotech",          title: "Incidencia hace 2 meses",  bio: "Refleja el ciclo generacional del vector. Aedes aegypti tarda ~10–14 días en completar su ciclo larva→adulto. Dos meses atrás captura la 'ola anterior' del brote." },
  incidencia_lag3:  { icon: "biotech",          title: "Incidencia hace 3 meses",  bio: "Patrón estacional trimestral. Tres meses atrás coincide con el inicio de la temporada de lluvias previa, que creó los criaderos que ahora alimentan el brote actual." },
  incidencia_lag4:  { icon: "history",          title: "Incidencia hace 4 meses",  bio: "Captura la inercia epidémica de mediano plazo. Útil para detectar si la región experimentó un brote temprano en el año que agotó la inmunidad de rebaño transitoria." },
  incidencia_lag5:  { icon: "history",          title: "Incidencia hace 5 meses",  bio: "Componente de memoria epidemiológica. Relacionado con la circulación de serotipos: un pico 5 meses atrás puede indicar qué serotipo viral dominó y la inmunidad residual actual." },
  incidencia_lag6:  { icon: "history",          title: "Incidencia hace 6 meses",  bio: "Semestre anterior. Permite al modelo distinguir entre un año hiperendémico (alta carga sostenida) y uno con brotes puntuales, influyendo en el pronóstico basal." },
  incidencia_lag7:  { icon: "history",          title: "Incidencia hace 7 meses",  bio: "Memoria epidémica de largo plazo. Ayuda a capturar el patrón bimodal del dengue (dos picos anuales) en regiones con dos temporadas de lluvias." },
  incidencia_lag8:  { icon: "history",          title: "Incidencia hace 8 meses",  bio: "Correlaciona con las condiciones climáticas del mismo mes del año anterior, capturando efectos estacionales recurrentes en el patrón de transmisión." },
  incidencia_lag9:  { icon: "history",          title: "Incidencia hace 9 meses",  bio: "Permite detectar ciclos interanuales: un brote intenso 9 meses atrás puede preceder un período de baja transmisión por agotamiento del huésped susceptible." },
  incidencia_lag10: { icon: "history",          title: "Incidencia hace 10 meses", bio: "Rezago de casi un año: captura el efecto de la estación equivalente del año anterior sobre la población susceptible actual." },
  incidencia_lag11: { icon: "history",          title: "Incidencia hace 11 meses", bio: "Comparación casi-anual. Permite al modelo comparar el mes actual con el mes equivalente del año pasado, detectando si la incidencia va al alza o a la baja interanualmente." },
  incidencia_lag12: { icon: "history",          title: "Incidencia hace 12 meses", bio: "El mismo mes del año anterior. Componente estacional pura: captura si esta época del año históricamente es de alto o bajo riesgo en este departamento específico." },

  // Variables climáticas base
  tmax_promedio:    { icon: "device_thermostat", title: "Temperatura máxima mensual",        bio: "Determina la tasa de replicación viral dentro del mosquito (período de incubación extrínseca). Entre 26–32°C el virus dengue se replica en ≈8–10 días; por debajo de 18°C la replicación se detiene." },
  tmin_promedio:    { icon: "device_thermostat", title: "Temperatura mínima mensual",        bio: "Regula la supervivencia nocturna del mosquito adulto. Noches cálidas (>20°C) permiten alimentación hematófaga continua y mayor oviposición, acelerando el ciclo reproductivo." },
  precipitacion:    { icon: "water_drop",        title: "Precipitación mensual acumulada",   bio: "La lluvia crea contenedores de agua estancada (floreros, llantas, recipientes) que son los principales criaderos de Aedes aegypti. Una lluvia intensa pero con drenaje insuficiente genera condiciones óptimas para la oviposición." },
  humedad_promedio: { icon: "humidity_mid",      title: "Humedad relativa mensual",          bio: "Humedad >60% extiende la vida media del mosquito adulto de ~2 semanas a más de 3 semanas. Además, facilita la búsqueda del huésped y aumenta la frecuencia de picadura." },
  agua_basica:      { icon: "water",             title: "Acceso a agua potable básica (%)",  bio: "Un acceso deficiente (<80%) obliga a almacenar agua en contenedores descubiertos en el hogar, que se convierten en criaderos primarios intradomiciliarios. SHAP negativo: mayor cobertura → menor riesgo de dengue." },
  densidad_poblacion: { icon: "group",           title: "Densidad de población (hab/km²)",   bio: "A mayor densidad, la distancia entre huésped y mosquito se reduce, aumentando la probabilidad de contacto. Las urbes densas amplifican los brotes por mayor disponibilidad de huéspedes susceptibles por unidad de área." },
  poblacion:        { icon: "group",             title: "Población total del departamento",  bio: "Controla el efecto de escala. Departamentos muy poblados tienen mayor número absoluto de casos incluso con la misma tasa de incidencia, lo que puede amplificar señales autoregresivas." },

  // Lags climáticos — temperatura máxima
  tmax_lag1: { icon: "device_thermostat", title: "Temperatura máxima hace 1 mes", bio: "Período de incubación extrínseca: el virus tarda 8–14 días en replicarse dentro del mosquito a 28°C. Un mes cálido antes implica que los vectores infectados ya están disponibles para transmitir ahora." },
  tmax_lag2: { icon: "device_thermostat", title: "Temperatura máxima hace 2 meses", bio: "Captura el efecto sobre la eclosión de huevos y desarrollo larval. A 28°C el ciclo larva→adulto es de ~7–10 días; con lag de 2 meses refleja la cohorte de adultos emergentes del mes anterior." },
  tmax_lag3: { icon: "device_thermostat", title: "Temperatura máxima hace 3 meses", bio: "Lag de mayor impacto entomológico: 3 meses atrás refleja la temperatura cuando comenzó el ciclo generacional completo (huevo→larva→pupa→adulto→infección→transmisión)." },
  tmax_lag4: { icon: "device_thermostat", title: "Temperatura máxima hace 4 meses", bio: "Efecto acumulado sobre la densidad vectorial. Cuatro meses de calor sostenido pueden producir múltiples generaciones de mosquitos, escalando la presión de transmisión." },
  tmax_lag5: { icon: "device_thermostat", title: "Temperatura máxima hace 5 meses", bio: "Refleja el inicio de una temporada calurosa extendida. Útil para detectar veranos largos donde el vector mantuvo alta densidad por muchos meses consecutivos." },
  tmax_lag6: { icon: "device_thermostat", title: "Temperatura máxima hace 6 meses", bio: "Correlación de medio año: captura si la región estuvo en verano/época seca vs. lluvias, condicionando el tipo de criaderos (artificiales vs. naturales) que predominan ahora." },

  // Lags climáticos — temperatura mínima
  tmin_lag1: { icon: "device_thermostat", title: "Temperatura mínima hace 1 mes", bio: "Indica si el mes pasado tuvo noches cálidas que permitieron alta actividad vectorial nocturna. Noches >18°C aceleran la oviposición y alimentación sanguínea del vector adulto." },
  tmin_lag2: { icon: "device_thermostat", title: "Temperatura mínima hace 2 meses", bio: "Meses con mínimas altas producen mayor supervivencia larval nocturna. Este lag captura la cohorte de mosquitos que emergió la semana pasada a partir de huevos depositados hace 2 meses." },
  tmin_lag3: { icon: "device_thermostat", title: "Temperatura mínima hace 3 meses", bio: "Temperatura nocturna trimestral: refleja si la región atravesó un período de noches frías que limitara la reproducción del vector, o si las noches se mantuvieron cálidas estimulando el ciclo." },
  tmin_lag4: { icon: "device_thermostat", title: "Temperatura mínima hace 4 meses", bio: "Largo plazo térmico nocturno. Cuatro meses de mínimas altas son un indicador de que el vector mantuvo alta densidad poblacional sostenida en ese período." },
  tmin_lag5: { icon: "device_thermostat", title: "Temperatura mínima hace 5 meses", bio: "Captura el perfil de temperatura nocturna de la temporada anterior. Relevante para regiones con marcada estacionalidad donde la temperatura nocturna define el inicio/fin de la temporada vectorial." },
  tmin_lag6: { icon: "device_thermostat", title: "Temperatura mínima hace 6 meses", bio: "Temperatura nocturna de hace 6 meses: marca si la región estaba en verano o invierno, condicionando la densidad vectorial basal de la temporada actual." },

  // Lags climáticos — precipitación
  precipitacion_lag1: { icon: "water_drop", title: "Precipitación hace 1 mes", bio: "La lluvia del mes pasado creó los criaderos que produjeron los adultos que están picando ahora. Lluvia intensa → más criaderos → más larvas → más adultos infectantes este mes." },
  precipitacion_lag2: { icon: "water_drop", title: "Precipitación hace 2 meses", bio: "Refleja los criaderos formados hace 2 meses. El ciclo larva→adulto es de ~7–14 días, por lo que la lluvia de hace 2 meses alimentó la cohorte de vectores que emergió el mes pasado." },
  precipitacion_lag3: { icon: "water_drop", title: "Precipitación hace 3 meses", bio: "Lag trimestral: captura si el inicio de la temporada de lluvias (3 meses atrás) fue intenso. El primer pulso de lluvia crea criaderos masivos que generan la primera ola de adultos 2–4 semanas después." },
  precipitacion_lag4: { icon: "water_drop", title: "Precipitación hace 4 meses", bio: "Precipitación de la temporada anterior. Relevante para regiones con estación lluviosa bien definida: permite al modelo saber si la carga vectorial acumulada fue alta o baja en este año." },
  precipitacion_lag5: { icon: "water_drop", title: "Precipitación hace 5 meses", bio: "Efecto de largo plazo: en regiones tropicales, lluvia abundante 5 meses atrás puede haber saturado el suelo y creado cuerpos de agua semi-permanentes que aún persisten como criaderos." },
  precipitacion_lag6: { icon: "water_drop", title: "Precipitación hace 6 meses", bio: "Precipitación del semestre anterior. Permite comparar si la región está en un año más lluvioso o seco que el promedio histórico, condicionando la disponibilidad de criaderos actuales." },

  // Lags climáticos — humedad
  humedad_lag1: { icon: "humidity_mid", title: "Humedad relativa hace 1 mes", bio: "Alta humedad el mes pasado extendió la vida del vector adulto, aumentando la probabilidad de que mosquitos infectados vivieran lo suficiente para transmitir el virus." },
  humedad_lag2: { icon: "humidity_mid", title: "Humedad relativa hace 2 meses", bio: "Captura el efecto de la humedad sobre la evaporación de criaderos: baja humedad hace 2 meses pudo haber secado recipientes con larvas, reduciendo la cohorte de adultos actuales." },
  humedad_lag3: { icon: "humidity_mid", title: "Humedad relativa hace 3 meses", bio: "Lag trimestral de humedad: identifica si la región atravesó una estación seca o húmeda que condicionó la disponibilidad de criaderos y la supervivencia del vector en ese período." },
  humedad_lag4: { icon: "humidity_mid", title: "Humedad relativa hace 4 meses", bio: "Indica el contexto climático de la temporada anterior. Alta humedad sostenida favorece la actividad crepuscular/nocturna del mosquito y la maduración de huevos de Aedes." },
  humedad_lag5: { icon: "humidity_mid", title: "Humedad relativa hace 5 meses", bio: "Efecto de largo plazo sobre la diapausa: huevos de Aedes pueden sobrevivir en ambientes secos hasta 12 meses. Alta humedad ahora puede activar huevos que la sequía de 5 meses atrás indujo a diapausa." },
  humedad_lag6: { icon: "humidity_mid", title: "Humedad relativa hace 6 meses", bio: "Perfil semestral de humedad: fundamental para regiones con dos estaciones marcadas. Permite al modelo detectar si la transición seco→húmedo ocurrió hace 6 meses, desencadenando el ciclo vectorial actual." },

  // Vecinos espaciales
  incidencia_vecinos_lag1: { icon: "location_on", title: "Incidencia vecinos (hace 1 mes)", bio: "Promedio de casos de los 3 departamentos geográficamente más cercanos el mes pasado. El dengue se difunde por contigüidad: viajeros, movilidad laboral y transporte terrestre entre departamentos vecinos." },
  incidencia_vecinos_lag2: { icon: "location_on", title: "Incidencia vecinos (hace 2 meses)", bio: "Difusión espacial retardada 2 meses: captura el 'frente de onda' epidémico que avanza desde focos vecinos. Un departamento puede prever su propio brote al observar el de sus vecinos." },
  incidencia_vecinos_lag3: { icon: "location_on", title: "Incidencia vecinos (hace 3 meses)", bio: "Onda epidémica trimestral: cuando los departamentos vecinos tuvieron un pico 3 meses atrás, la presión migratoria de casos importados llega ahora al departamento objetivo." },
  incidencia_vecinos_lag4: { icon: "location_on", title: "Incidencia vecinos (hace 4 meses)", bio: "Difusión regional de mediano plazo. Relevante para cuencas amazónicas o corredores viales donde el virus viaja lentamente entre poblaciones rurales dispersas." },
  incidencia_vecinos_lag5: { icon: "location_on", title: "Incidencia vecinos (hace 5 meses)", bio: "Captura la propagación desde focos endémicos lejanos. Departamentos con alta endemia vecina hace 5 meses pueden estar exportando casos con serotipos específicos." },
  incidencia_vecinos_lag6: { icon: "location_on", title: "Incidencia vecinos (hace 6 meses)", bio: "Presión vectorial regional de hace 6 meses: cuando los vecinos tuvieron un semestre de alta incidencia, el pool de mosquitos infectados en la región geográfica ampliada es mayor." },

  // Medias móviles
  incidencia_roll3:  { icon: "area_chart",   title: "Media móvil 3 meses",  bio: "Promedio de incidencia de los últimos 3 meses: suaviza picos puntuales y revela la tendencia de corto plazo. Un roll3 alto indica que el brote no fue un evento aislado sino una tendencia sostenida." },
  incidencia_roll6:  { icon: "area_chart",   title: "Media móvil 6 meses",  bio: "Tendencia semestral de incidencia: permite al modelo distinguir entre departamentos crónicamente endémicos (roll6 alto permanente) y los que experimentan brotes esporádicos." },
  incidencia_roll12: { icon: "area_chart",   title: "Media móvil 12 meses", bio: "Carga anual de dengue: el promedio de los últimos 12 meses es el indicador de endemicidad más robusto. Departamentos con roll12 alto tienen vectores bien establecidos y población con inmunidad heterogénea." },

  // Variables derivadas epidemiológicas
  amplitud_termica:       { icon: "device_thermostat", title: "Amplitud térmica (tmax − tmin)", bio: "La diferencia entre temperatura máxima y mínima diaria. Rangos amplios (>10°C) reducen la actividad vectorial porque las noches frías matan los adultos. Rangos estrechos en climas húmedos favorecen al mosquito." },
  temperatura_media:      { icon: "device_thermostat", title: "Temperatura media mensual",       bio: "Promedio de tmax y tmin. En el rango 25–30°C, la temperatura media óptima para Aedes aegypti maximiza la tasa de picadura, la oviposición y la replicación viral dentro del vector." },
  precipitacion_anomalia: { icon: "water_drop",        title: "Anomalía de precipitación",       bio: "Desviación de la lluvia mensual respecto a la media histórica del mismo mes. Lluvias anómalas (positivas) crean criaderos inusuales; anomalías negativas (sequías) pueden concentrar larvas en contenedores residuales." },
  aceleracion_incidencia: { icon: "trending_up",       title: "Aceleración de incidencia",       bio: "Segunda derivada temporal: mide si la incidencia está acelerando (segunda derivada positiva) o desacelerando. Una aceleración positiva alerta de un brote en fase exponencial temprana." },
  cambio_interanual:      { icon: "trending_up",       title: "Cambio interanual de incidencia", bio: "Diferencia entre la incidencia actual y la del mismo mes del año anterior. Permite detectar si 2022 está siendo más intenso que 2021, incluso si ambos tienen incidencia moderada en términos absolutos." },
  tendencia_1m:           { icon: "show_chart",        title: "Tendencia de 1 mes",              bio: "Diferencia entre incidencia del mes actual y el anterior (primera derivada). Pendiente positiva indica brote en crecimiento; negativa indica descenso. El Agente 6 usa esto para detectar el régimen epidémico." },
  tendencia_3m:           { icon: "show_chart",        title: "Tendencia de 3 meses",            bio: "Diferencia entre incidencia actual y la de hace 3 meses. Captura tendencias de mediano plazo, filtrando variaciones mensuales ruidosas para revelar si el departamento está en ciclo ascendente o descendente." },
  fase_ascendente:        { icon: "trending_up",       title: "Fase ascendente (binario)",       bio: "Vale 1 si la incidencia actual es mayor a la de los 3 meses previos (tendencia al alza). Señal de alerta temprana de brote: el modelo XGBoost le da más peso a las variables autorregresivas cuando esta señal está activa." },
  indicador_brote:        { icon: "crisis_alert",      title: "Indicador de brote activo (binario)", bio: "Vale 1 si la incidencia supera el percentil 90 histórico del departamento (umbral de epidemia local). Activa el modo de alta alerta en el Agente 6 y redirige los pesos del ensemble hacia el LSTM." },

  // Indicadores de contexto
  indicador_covid: { icon: "coronavirus",  title: "Período COVID-19 (2020-2021)", bio: "Las medidas de confinamiento redujeron la movilidad humana pero también la vigilancia epidemiológica. Controla el sub-reporte de casos en 2020-2021 y los efectos indirectos sobre los patrones de transmisión." },
  indicador_nino:  { icon: "waves",        title: "Evento El Niño",               bio: "El Niño causa calentamiento y sequías en regiones andinas latinoamericanas, pero lluvias intensas en el Pacífico. En algunas regiones amplifica el dengue (más criaderos); en otras lo reduce (sequía elimina larvas)." },
  indicador_nina:  { icon: "waves",        title: "Evento La Niña",               bio: "La Niña genera lluvias intensas en regiones andinas y amazónicas, creando condiciones óptimas para Aedes aegypti. Históricamente los brotes de dengue más severos en Perú, Colombia y Ecuador coinciden con La Niña." },

  // Codificación cíclica
  mes_sin: { icon: "calendar_month", title: "Codificación cíclica del mes (seno)",   bio: "sin(2π·mes/12) — componente sinusoidal que representa la posición del mes en el ciclo anual. Permite al modelo aprender que diciembre y enero están temporalmente cercanos (no separados por 11 meses)." },
  mes_cos: { icon: "calendar_month", title: "Codificación cíclica del mes (coseno)", bio: "cos(2π·mes/12) — componente coseno complementaria. Junto con mes_sin forma un vector unitario en el círculo trigonométrico que representa unívocamente cada mes sin discontinuidades en el año." },

  // Dummies de país
  pais_ARG: { icon: "flag", title: "Dummy: Argentina", bio: "Controla el patrón epidemiológico específico de Argentina: dengue estacional concentrado en verano austral (nov–mar), principalmente en el norte. Los umbrales climáticos del vector difieren de países tropicales." },
  pais_BOL: { icon: "flag", title: "Dummy: Bolivia",   bio: "Ajusta por el contexto epidemiológico boliviano: regiones bajas (Beni, Santa Cruz) con transmisión endémica vs. altiplano sin transmisión. Controla las diferencias en sistema de vigilancia y sub-reporte." },
  pais_BRA: { icon: "flag", title: "Dummy: Brasil",    bio: "Brasil concentra el 70–80% de los casos de América Latina. Esta dummy ajusta por el mayor volumen de casos, la mayor capacidad de vigilancia y los patrones de circulación de múltiples serotipos simultáneamente." },
  pais_COL: { icon: "flag", title: "Dummy: Colombia",  bio: "Ajusta por el patrón bimodal de Colombia (dos picos anuales: abril-junio y octubre-noviembre) asociado a dos temporadas de lluvias. Controla también la diversidad altitudinal entre costa, valles y Amazonía." },
  pais_ECU: { icon: "flag", title: "Dummy: Ecuador",   bio: "Controla el contexto ecuatoriano: la costa del Pacífico es hiperendémica mientras que la Sierra andina tiene transmisión limitada. Ajusta por el sistema de vigilancia del SIVE-Alerta del MSP Ecuador." },
  pais_MEX: { icon: "flag", title: "Dummy: México",    bio: "Ajusta por la diversidad climática mexicana (Pacífico, Golfo, interior seco). México tiene uno de los sistemas de vigilancia más robustos de la región, con menor sub-reporte relativo." },
  pais_PAN: { icon: "flag", title: "Dummy: Panamá",    bio: "Ajusta por el contexto de Panamá como corredor de tránsito continental con alta movilidad y climas tropicales uniformes. La circulación de los cuatro serotipos DENV 1-4 es simultánea." },
  pais_PER: { icon: "flag", title: "Dummy: Perú",      bio: "Ajusta por el patrón peruano: la selva amazónica concentra la mayoría de casos con brotes asociados a El Niño Costero. La región amazónica mantiene alta endemia mientras Lima y la sierra permanecen libres." },
};

function getShapBio(featureName) {
  if (SHAP_BIO[featureName]) return SHAP_BIO[featureName];
  const prefixMap = [
    ["incidencia_lag",        "biotech",          "Incidencia de dengue con rezago",  "Casos de dengue reportados k meses antes del período de predicción. Las series autorregresivas capturan el momentum epidémico: si hay casos ahora, habrá casos próximamente."],
    ["tmax_lag",              "device_thermostat","Temperatura máxima con rezago",     "Temperatura máxima mensual k meses antes. Determina la velocidad de replicación viral dentro del vector y el tiempo de incubación extrínseca del dengue."],
    ["tmin_lag",              "device_thermostat","Temperatura mínima con rezago",     "Temperatura mínima mensual k meses antes. Controla la supervivencia nocturna del mosquito Aedes aegypti y la viabilidad de las larvas."],
    ["precipitacion_lag",     "water_drop",       "Precipitación con rezago",          "Lluvia acumulada k meses antes. Genera los criaderos de agua estancada donde se reproducen las larvas del vector Aedes aegypti."],
    ["humedad_lag",           "humidity_mid",     "Humedad relativa con rezago",       "Humedad relativa k meses antes. La humedad elevada extiende la vida del mosquito adulto y su radio de vuelo efectivo."],
    ["incidencia_vecinos_lag","location_on",      "Incidencia vecinal con rezago",     "Promedio de incidencia de los 3 departamentos geográficamente más cercanos, k meses antes. Captura la difusión espacial del dengue a través de la movilidad humana regional."],
    ["pais_",                 "flag",             "Variable indicadora de país",       "Variable binaria que identifica el país. Controla diferencias estructurales entre sistemas de vigilancia epidemiológica, densidad vectorial endémica y contexto sociosanitario."],
  ];
  for (const [prefix, icon, title, bio] of prefixMap) {
    if (featureName.startsWith(prefix)) return { icon, title, bio };
  }
  return null;
}

function ShapTooltip({ feature, children }) {
  const [show, setShow] = React.useState(false);
  const info = getShapBio(feature);
  if (!info) return <>{children}</>;
  return (
    <div
      className="relative inline-flex items-center gap-xs cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      <span className="material-symbols-outlined text-[14px] text-on-surface-variant/40 hover:text-primary transition-colors">info</span>
      {show && (
        <div
          className="absolute z-50 bottom-full left-0 mb-2 w-72 rounded-xl shadow-2xl border border-outline-variant animate-fade-in pointer-events-none"
          style={{ background: "rgb(var(--color-surface-container-high))" }}
        >
          <div className="flex items-center gap-xs px-md pt-md pb-xs border-b border-outline-variant/40">
            <span className="material-symbols-outlined text-[16px] text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>{info.icon}</span>
            <span className="text-[12px] font-bold text-primary leading-snug">{info.title}</span>
          </div>
          <p className="px-md py-sm text-[11px] text-on-surface-variant leading-relaxed">{info.bio}</p>
        </div>
      )}
    </div>
  );
}

const MOCK_SHAP_GLOBAL = [
  { feature: "incidencia_lag1", importance: 0.285 },
  { feature: "tmax_promedio", importance: 0.182 },
  { feature: "precipitacion", importance: 0.144 },
  { feature: "humedad_promedio", importance: 0.095 },
  { feature: "agua_basica", importance: 0.078 },
  { feature: "incidencia_vecinos_lag1", importance: 0.065 },
  { feature: "densidad_poblacion", importance: 0.042 },
  { feature: "tmin_promedio", importance: 0.038 },
  { feature: "tmax_lag1", importance: 0.031 },
  { feature: "precipitacion_lag1", importance: 0.024 },
];

const MONTH_NAMES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"];

export default function ExplainabilityView({ activeSubtab, simulationHistory = [], onClearHistory }) {
  // ─── Global SHAP state ───
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recalculating, setRecalculating] = useState(false);
  const [shapPage, setShapPage] = useState(0);
  const SHAP_PAGE_SIZE = 15;

  // ─── Local SHAP state ───
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [localResult, setLocalResult] = useState(null);

  const lastSimulation = simulationHistory[selectedIdx] ?? null;

  // Reset result when selected simulation changes
  useEffect(() => { setLocalResult(null); setLocalError(null); }, [selectedIdx]);
  // Always point to newest when a new simulation arrives
  useEffect(() => { setSelectedIdx(0); }, [simulationHistory.length]);

  // ─── Fetch global SHAP ───
  const fetchShap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/explain/global`);
      if (!response.ok) throw new Error("No se pudo cargar la explicabilidad SHAP");
      const raw = await response.json();
      const arr = Object.entries(raw)
        .map(([feature, importance]) => ({ feature, importance }))
        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
      setShapData(arr);
    } catch (err) {
      console.warn("Backend explain offline, usando datos demo...", err.message);
      setShapData(MOCK_SHAP_GLOBAL);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchShap(); }, [fetchShap]);

  const handleAnalyzeLocal = useCallback(async () => {
    if (!lastSimulation) return;
    setLocalLoading(true);
    setLocalError(null);
    setLocalResult(null);
    try {
      const res = await fetch(`${API_URL}/api/predict/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          iso_a0: lastSimulation.iso_a0,
          adm_1_name: lastSimulation.adm_1_name,
          mes: lastSimulation.mes,
          clima_overrides: lastSimulation.clima_overrides, // ya viene sin mes_sin/mes_cos/rolls
          include_shap: true,
        }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      if (!data.shap_local) throw new Error("El backend no retornó valores SHAP locales.");
      const shapArr = Object.entries(data.shap_local)
        .map(([feature, value]) => ({ feature, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 12);
      setLocalResult({
        prediction: data.prediccion_ensemble ?? data.prediccion_ml,
        riesgo: data.riesgo_ensemble ?? data.riesgo_ml,
        shapArr,
      });
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setLocalLoading(false);
    }
  }, [lastSimulation]);

  const handleRecalculate = async () => {
    setRecalculating(true);
    await fetchShap();
    setRecalculating(false);
  };

  const handleExport = () => {
    const doc = new jsPDF();
    const fecha = new Date().toLocaleDateString("es-ES", { year: "numeric", month: "long", day: "numeric" });
    const PRIMARY = [30, 58, 95];
    const GRAY    = [100, 100, 100];

    doc.setFontSize(18);
    doc.setTextColor(...PRIMARY);
    doc.text("DenguePredict — Explicabilidad SHAP", 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(...GRAY);
    doc.text(`Agente 3 · TreeSHAP | Generado: ${fecha}`, 14, 26);

    if (activeSubtab === "Local SHAP") {
      if (!localResult) {
        alert("Primero ejecuta una simulación en el Predictor y luego haz clic en 'Explicar simulación'.");
        return;
      }
      const deptLabel = lastSimulation ? `${lastSimulation.adm_1_name} (${lastSimulation.country})` : "—";
      doc.setFontSize(13);
      doc.setTextColor(...PRIMARY);
      doc.text(`SHAP Local — ${deptLabel}`, 14, 36);
      doc.setFontSize(10);
      doc.setTextColor(...GRAY);
      doc.text(
        `Predicción Ensemble: ${localResult.prediction?.toFixed(2)} casos/100k · Riesgo: ${localResult.riesgo?.nivel ?? "—"}`,
        14, 43
      );
      autoTable(doc, {
        startY: 48,
        head: [["Variable", "Valor SHAP", "Dirección"]],
        body: localResult.shapArr.map((f) => [
          f.feature,
          (f.value > 0 ? "+" : "") + f.value.toFixed(6),
          f.value >= 0 ? "Aumenta riesgo" : "Reduce riesgo",
        ]),
        headStyles: { fillColor: PRIMARY },
        alternateRowStyles: { fillColor: [245, 248, 255] },
        columnStyles: {
          1: { halign: "right", fontStyle: "bold" },
          2: { halign: "center" },
        },
      });
    } else {
      // Global SHAP
      if (!shapData || shapData.length === 0) {
        alert("Los datos SHAP globales aún están cargando.");
        return;
      }
      doc.setFontSize(13);
      doc.setTextColor(...PRIMARY);
      doc.text("SHAP Global — Importancia Media de Variables (todos los departamentos)", 14, 36);
      autoTable(doc, {
        startY: 42,
        head: [["Ranking", "Variable", "Importancia SHAP media", "Dirección"]],
        body: shapData.map((f, i) => [
          `#${i + 1}`,
          f.feature,
          f.importance.toFixed(6),
          f.importance >= 0 ? "Aumenta riesgo" : "Reduce riesgo",
        ]),
        headStyles: { fillColor: PRIMARY },
        alternateRowStyles: { fillColor: [245, 248, 255] },
        columnStyles: {
          2: { halign: "right", fontStyle: "bold" },
          3: { halign: "center" },
        },
      });
    }

    const pageH = doc.internal.pageSize.getHeight();
    doc.setFontSize(8);
    doc.setTextColor(...GRAY);
    doc.text("DenguePredict — Proyecto Final FISI-UNMSM | Uso académico", 14, pageH - 8);

    const filename = activeSubtab === "Local SHAP"
      ? `SHAP_Local_${localDept.replace(/\s+/g, "_")}.pdf`
      : "SHAP_Global_DenguePredict.pdf";
    doc.save(filename);
  };

  const maxVal = Array.isArray(shapData) && shapData.length > 0 
    ? Math.max(...shapData.map((f) => Math.abs(f.importance))) 
    : 1;

  // Timestamp for display
  const now = new Date();
  const hours12 = now.getHours() % 12 || 12;
  const ampm = now.getHours() >= 12 ? "PM" : "AM";
  const timeStr = `Hoy, ${hours12.toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")} ${ampm}`;

  return (
    <div className="max-w-[1440px] mx-auto text-on-surface">
      {/* Header Section */}
      <div className="mb-lg flex flex-col md:flex-row md:items-end justify-between gap-md">
        <div>
          <h1 className="text-headline-lg text-primary font-bold mb-xs">
            Módulo de Explicabilidad Local y Global (SHAP)
          </h1>
          <div className="flex items-center gap-sm flex-wrap">
            <span className="text-on-surface-variant text-label-md">Última actualización: {timeStr}</span>
          </div>
        </div>
        <div className="flex gap-md">
          <button 
            onClick={handleExport}
            className="px-md py-sm border border-primary text-primary rounded-lg text-label-md font-medium hover:bg-primary/5 transition-colors flex items-center gap-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">download</span> Exportar PDF
          </button>
          <button 
            onClick={handleRecalculate}
            disabled={recalculating}
            className="px-md py-sm bg-primary text-on-primary rounded-lg text-label-md font-medium hover:bg-primary-container transition-colors flex items-center gap-sm cursor-pointer disabled:opacity-55"
          >
            <span className="material-symbols-outlined text-[18px] animate-pulse">refresh</span> 
            {recalculating ? "Procesando..." : "Re-calcular SHAP"}
          </button>
        </div>
      </div>

      {/* Bento Grid */}
      <div className="grid grid-cols-1 gap-lg mb-lg">

        {/* ═══ TAB CONTENT: GLOBAL SHAP Summary Plot ═══ */}
        {activeSubtab === "Global SHAP" && (
          <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] max-w-4xl mx-auto w-full animate-fade-in">
            <div className="flex items-center justify-between mb-lg">
              <h3 className="text-headline-md text-on-surface font-bold">SHAP Global Summary Plot</h3>
              <span
                className="material-symbols-outlined text-on-surface-variant cursor-help"
                title="Impacto global de las variables en el modelo a nivel de todo el continente"
              >
                info
              </span>
            </div>
            <p className="text-on-surface-variant text-label-md mb-xl">
              Importancia media de las características (Magnitud del valor SHAP promedio de los agentes ML/DL)
            </p>

            {/* Loading */}
            {loading && (
              <div className="space-y-lg">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="space-y-sm">
                    <div className="flex justify-between">
                      <div className="h-4 w-28 shimmer rounded"></div>
                      <div className="h-4 w-12 shimmer rounded"></div>
                    </div>
                    <div className="w-full h-4 shimmer rounded-full"></div>
                  </div>
                ))}
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="bg-error-container p-md rounded-lg flex items-center gap-md">
                <span className="material-symbols-outlined text-on-error-container">error</span>
                <div>
                  <p className="text-label-md text-on-error-container font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* SHAP Bars */}
            {Array.isArray(shapData) && shapData.length > 0 && (() => {
              const totalPages = Math.ceil(shapData.length / SHAP_PAGE_SIZE);
              const pageData  = shapData.slice(shapPage * SHAP_PAGE_SIZE, (shapPage + 1) * SHAP_PAGE_SIZE);
              return (
                <div className="space-y-lg">
                  {pageData.map((feature, i) => {
                    const globalRank = shapPage * SHAP_PAGE_SIZE + i + 1;
                    const pct = (Math.abs(feature.importance) / maxVal) * 100;
                    return (
                      <div key={feature.feature} className="space-y-xs group/bar">
                        <div className="flex justify-between items-center" style={{ fontVariantNumeric: "tabular-nums" }}>
                          <div className="flex items-center gap-sm">
                            <span className="text-[10px] font-bold text-on-surface-variant/50 w-5 text-right flex-shrink-0">#{globalRank}</span>
                            <ShapTooltip feature={feature.feature}>
                              <span className="text-label-md text-on-surface font-medium group-hover/bar:text-primary transition-colors">{feature.feature}</span>
                            </ShapTooltip>
                          </div>
                          <span className="text-label-md text-on-surface-variant font-mono">{Math.abs(feature.importance).toFixed(4)}</span>
                        </div>
                        <div className="w-full bg-surface-container-low dark:bg-zinc-800 h-3 rounded-full overflow-hidden">
                          <div
                            className="chart-bar h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all duration-300 group-hover/bar:brightness-110"
                            style={{ width: `${Math.max(pct, 2)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between pt-lg border-t border-outline-variant mt-lg">
                      <button
                        onClick={() => setShapPage(p => p - 1)}
                        disabled={shapPage === 0}
                        className="flex items-center gap-xs text-label-md text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[18px]">chevron_left</span>
                        Anterior
                      </button>
                      <div className="flex items-center gap-xs">
                        {[...Array(totalPages)].map((_, pi) => (
                          <button
                            key={pi}
                            onClick={() => setShapPage(pi)}
                            className={`w-7 h-7 rounded-lg text-[12px] font-bold transition-colors cursor-pointer
                              ${pi === shapPage ? "bg-primary text-on-primary" : "text-on-surface-variant hover:bg-surface-container"}`}
                          >
                            {pi + 1}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={() => setShapPage(p => p + 1)}
                        disabled={shapPage >= totalPages - 1}
                        className="flex items-center gap-xs text-label-md text-primary disabled:opacity-30 disabled:cursor-not-allowed hover:bg-surface-container px-sm py-xs rounded-lg transition-colors cursor-pointer"
                      >
                        Siguiente
                        <span className="material-symbols-outlined text-[18px]">chevron_right</span>
                      </button>
                    </div>
                  )}

                  {/* Legend */}
                  <div className="flex justify-between items-center border-t border-outline-variant pt-md">
                    <div className="flex items-center gap-sm">
                      <div className="w-3 h-3 rounded-full bg-gradient-to-r from-primary to-secondary" />
                      <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Mayor barra = mayor influencia en la predicción</span>
                    </div>
                    <span className="text-[11px] text-on-surface-variant/50">{shapData.length} variables · pág. {shapPage + 1}/{totalPages}</span>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* ═══ TAB CONTENT: LOCAL SHAP ═══ */}
        {activeSubtab === "Local SHAP" && (
          <div className="max-w-4xl mx-auto w-full flex flex-col gap-lg animate-fade-in">

            {/* ── Historial de simulaciones ── */}
            <div className="bg-white dark:bg-zinc-900 border border-outline-variant rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] overflow-hidden">
              <div className="flex items-center justify-between px-lg py-md border-b border-outline-variant">
                <div>
                  <h3 className="text-headline-md text-on-surface font-bold">Historial de Simulaciones</h3>
                  <p className="text-label-md text-on-surface-variant mt-xs">
                    Selecciona una simulación para explicar con SHAP
                  </p>
                </div>
                {simulationHistory.length > 0 && (
                  <button
                    onClick={onClearHistory}
                    className="text-label-md text-error hover:underline flex items-center gap-xs cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[16px]">delete_sweep</span> Limpiar
                  </button>
                )}
              </div>

              {simulationHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-xl text-on-surface-variant gap-md">
                  <span className="material-symbols-outlined text-[40px] opacity-30">sensors</span>
                  <p className="text-label-md text-center">
                    Primero ejecuta una simulación en el <strong className="text-primary">Predictor</strong> y vuelve aquí para explicarla.
                  </p>
                </div>
              ) : (
                <ul className="divide-y divide-outline-variant max-h-64 overflow-y-auto">
                  {simulationHistory.map((sim, idx) => {
                    const isSelected = idx === selectedIdx;
                    const timeLabel = sim.timestamp
                      ? sim.timestamp.toLocaleTimeString("es-PE", { hour: "2-digit", minute: "2-digit" })
                      : "";
                    return (
                      <li
                        key={sim.id}
                        onClick={() => setSelectedIdx(idx)}
                        className={`flex items-center gap-md px-lg py-sm cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-primary/8 border-l-4 border-l-primary"
                            : "hover:bg-surface-container-low border-l-4 border-l-transparent"
                        }`}
                      >
                        <span className={`material-symbols-outlined text-[20px] ${isSelected ? "text-primary" : "text-on-surface-variant"}`}>
                          {isSelected ? "radio_button_checked" : "radio_button_unchecked"}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className={`text-label-md font-bold truncate ${isSelected ? "text-primary" : "text-on-surface"}`}>
                            {sim.adm_1_name} <span className="font-normal text-on-surface-variant">({sim.country})</span>
                          </p>
                          <p className="text-[12px] text-on-surface-variant">
                            Mes: <strong>{MONTH_NAMES[(sim.mes ?? 1) - 1]}</strong> · {Object.keys(sim.clima_overrides ?? {}).length} variables
                          </p>
                        </div>
                        <div className="flex flex-col items-end gap-xs flex-shrink-0">
                          {idx === 0 && (
                            <span className="px-xs py-[2px] bg-primary/10 text-primary text-[10px] font-bold rounded-full">ÚLTIMA</span>
                          )}
                          <span className="text-[11px] text-on-surface-variant">{timeLabel}</span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            {/* ── Panel de explicación ── */}
            {lastSimulation && (
              <div className="bg-white dark:bg-zinc-900 border border-outline-variant p-lg rounded-xl shadow-[0px_4px_20px_rgba(30,58,95,0.04)] flex flex-col">
                <div className="mb-lg">
                  <h3 className="text-headline-md text-on-surface font-bold">SHAP Local — Explicación de Simulación</h3>
                  <p className="text-label-md text-on-surface-variant mt-xs">
                    Descompone qué variables contribuyeron más a la predicción y en qué dirección.
                  </p>
                </div>

                {/* Contexto + botón */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-md mb-lg p-md rounded-xl border border-outline-variant bg-surface-container-low">
                  <div className="flex-1 space-y-xs">
                    <p className="text-label-md font-bold text-on-surface">
                      {lastSimulation.country} · {lastSimulation.adm_1_name}
                    </p>
                    <p className="text-[12px] text-on-surface-variant">
                      Mes objetivo: <strong>{MONTH_NAMES[(lastSimulation.mes ?? 1) - 1]}</strong> · {Object.keys(lastSimulation.clima_overrides ?? {}).length} variables configuradas
                    </p>
                  </div>
                  <button
                    onClick={handleAnalyzeLocal}
                    disabled={localLoading}
                    className="px-lg py-sm bg-primary text-on-primary rounded-lg text-label-md font-bold hover:bg-primary/90 transition-colors flex items-center gap-sm cursor-pointer disabled:opacity-55 whitespace-nowrap"
                  >
                    {localLoading
                      ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span> Analizando...</>
                      : <><span className="material-symbols-outlined text-[16px]">analytics</span> Explicar simulación</>}
                  </button>
                </div>

                {/* Error */}
                {localError && (
                  <div className="bg-error-container p-md rounded-lg flex items-center gap-md mb-md">
                    <span className="material-symbols-outlined text-on-error-container">error</span>
                    <p className="text-label-md text-on-error-container font-medium">{localError}</p>
                  </div>
                )}

                {/* Empty state */}
                {!localResult && !localLoading && !localError && (
                  <div className="flex flex-col items-center justify-center py-lg text-on-surface-variant gap-md">
                    <span className="material-symbols-outlined text-[48px] opacity-30">bar_chart_4_bars</span>
                    <p className="text-label-md">Haz clic en "Explicar simulación" para analizar</p>
                  </div>
                )}

                {/* Results */}
                {localResult && (
                  <div className="animate-fade-in">
                    <div className="flex items-center gap-lg mb-lg p-md rounded-lg border border-outline-variant bg-surface-container-low">
                      <div>
                        <p className="text-label-md text-on-surface-variant">Predicción Ensemble</p>
                        <p className="text-headline-lg font-bold text-primary" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {localResult.prediction?.toFixed(1)} <span className="text-label-md font-normal opacity-60">casos/100k</span>
                        </p>
                      </div>
                      <div
                        className="px-md py-xs rounded-full text-label-md font-bold text-white"
                        style={{ backgroundColor: localResult.riesgo?.color ?? "#10b981" }}
                      >
                        {localResult.riesgo?.nivel ?? "—"}
                      </div>
                    </div>

                    <p className="text-label-md font-bold text-on-surface-variant uppercase tracking-wider mb-md">
                      Top {localResult.shapArr.length} variables por impacto SHAP
                    </p>
                    <div className="space-y-md">
                      {(() => {
                        const maxAbs = Math.max(...localResult.shapArr.map((f) => Math.abs(f.value)), 0.001);
                        return localResult.shapArr.map((feat) => {
                          const pct = (Math.abs(feat.value) / maxAbs) * 100;
                          const isNeg = feat.value < 0;
                          return (
                            <div key={feat.feature} className="space-y-xs">
                              <div className="flex justify-between items-center" style={{ fontVariantNumeric: "tabular-nums" }}>
                                <ShapTooltip feature={feat.feature}>
                                  <span className="text-label-md text-on-surface font-medium">{feat.feature}</span>
                                </ShapTooltip>
                                <span className={`text-label-md font-bold ${isNeg ? "text-blue-600" : "text-orange-500"}`}>
                                  {feat.value > 0 ? "+" : ""}{feat.value.toFixed(4)}
                                </span>
                              </div>
                              <div className="w-full bg-surface-container-low h-3.5 rounded-full overflow-hidden">
                                <div
                                  className={`chart-bar h-full rounded-full bg-gradient-to-r ${isNeg ? "from-blue-600 to-blue-400" : "from-orange-400 to-orange-600"}`}
                                  style={{ width: `${Math.max(pct, 3)}%` }}
                                ></div>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>

                    <div className="mt-lg flex justify-between items-center border-t border-outline-variant pt-md">
                      <div className="flex items-center gap-sm">
                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                        <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Reduce riesgo (SHAP &lt; 0)</span>
                      </div>
                      <div className="flex items-center gap-sm">
                        <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Aumenta riesgo (SHAP &gt; 0)</span>
                        <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ═══ Bottom: Scientific Interpretation ═══ */}
      <div className="bg-[#eff6ff] dark:bg-sky-950/20 border border-outline-variant p-lg rounded-xl flex gap-lg items-start mb-lg max-w-4xl mx-auto w-full">
        <div className="p-sm bg-white dark:bg-zinc-800 rounded-lg shadow-sm flex-shrink-0">
          <span className="material-symbols-outlined text-primary text-[32px]">lightbulb</span>
        </div>
        <div>
          <h4 className="text-headline-md text-on-surface font-bold mb-sm">
            Interpretación Científica de Coeficientes Shapley
          </h4>
          <div className="space-y-md text-on-surface-variant text-body-md leading-relaxed">
            <p>
              Los valores SHAP (SHapley Additive exPlanations) descomponen la predicción final en la contribución
              individual de cada variable climática y epidemiológica. Para la toma de decisiones en salud pública,
              un valor positivo indica un <strong className="text-on-surface">aumento del riesgo relativo</strong>,
              mientras que un valor negativo sugiere <strong className="text-on-surface">atenuación de la incidencia</strong>.
            </p>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-md">
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>incidencia_lag1:</strong> El historial inmediato de casos sigue siendo el predictor más
                  robusto de brotes epidémicos.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>tmax_promedio:</strong> Las temperaturas máximas extremas aceleran el ciclo metabólico del
                  vector <em>Aedes aegypti</em>.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>precipitación:</strong> Los volúmenes de lluvia crean hábitats de criaderos acuáticos
                  propicios para el vector.
                </span>
              </li>
              <li className="flex items-start gap-sm">
                <span className="material-symbols-outlined text-primary text-[18px] mt-1">check_circle</span>
                <span>
                  <strong>humedad_relativa:</strong> La humedad elevada extiende la supervivencia del mosquito
                  adulto y su capacidad de vuelo.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
