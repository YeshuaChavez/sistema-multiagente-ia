"""
AGENTE 1 — Recolección de Datos Climáticos
NASA POWER API — 37 países de Latinoamérica y el Caribe

Descarga temperatura máxima, temperatura mínima, precipitación
y humedad relativa para la capital de cada país LATAM.
Agrega directamente a mensual — NO guarda diario.

Uso:
    pip install requests pandas
    python agente1_nasa_power_latam.py

Output:
    nasa_power_latam_mensual.csv  → datos mensuales por país (2000-2024)
"""

import requests
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────────────────
# 37 PAÍSES LATAM del dataset OpenDengue con coordenadas de su capital
# ─────────────────────────────────────────────────────────────────────────────
PAISES = {
    "ARG": {"nombre": "ARGENTINA",                          "lat": -34.61, "lon": -58.38},
    "BOL": {"nombre": "BOLIVIA",                            "lat": -17.78, "lon": -63.18}, # Santa Cruz de la Sierra (tropical lowlands)
    "BRA": {"nombre": "BRAZIL",                             "lat": -15.78, "lon": -47.93},
    "CHL": {"nombre": "CHILE",                              "lat": -33.46, "lon": -70.65},
    "COL": {"nombre": "COLOMBIA",                           "lat":  10.96, "lon": -74.79}, # Barranquilla (tropical sea level)
    "CRI": {"nombre": "COSTA RICA",                         "lat":   9.93, "lon": -84.08},
    "CUB": {"nombre": "CUBA",                               "lat":  23.13, "lon": -82.38},
    "DOM": {"nombre": "DOMINICAN REPUBLIC",                 "lat":  18.48, "lon": -69.90},
    "ECU": {"nombre": "ECUADOR",                            "lat":  -2.17, "lon": -79.92}, # Guayaquil (tropical sea level)
    "GTM": {"nombre": "GUATEMALA",                          "lat":  14.64, "lon": -90.51},
    "HTI": {"nombre": "HAITI",                              "lat":  18.54, "lon": -72.34},
    "HND": {"nombre": "HONDURAS",                           "lat":  14.10, "lon": -87.22},
    "MEX": {"nombre": "MEXICO",                             "lat":  19.17, "lon": -96.13}, # Veracruz (tropical coast)
    "NIC": {"nombre": "NICARAGUA",                          "lat":  12.13, "lon": -86.28},
    "PAN": {"nombre": "PANAMA",                             "lat":   8.99, "lon": -79.52},
    "PER": {"nombre": "PERU",                               "lat":  -5.19, "lon": -80.63}, # Piura (highly endemic tropical area)
    "PRY": {"nombre": "PARAGUAY",                           "lat": -25.28, "lon": -57.65},
    "SLV": {"nombre": "EL SALVADOR",                        "lat":  13.69, "lon": -89.19},
    "URY": {"nombre": "URUGUAY",                            "lat": -34.86, "lon": -56.17},
    "VEN": {"nombre": "VENEZUELA",                          "lat":  10.49, "lon": -66.88},
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN NASA POWER — endpoint mensual directo
# ─────────────────────────────────────────────────────────────────────────────
PARAMETROS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"
BASE_URL   = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def descargar_mensual_pais(iso, nombre, lat, lon):
    """Descarga datos directamente en resolución mensual desde NASA POWER."""
    params = {
        "parameters": PARAMETROS,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      "2000",
        "end":        "2024",
        "format":     "JSON",
    }
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()

    data  = resp.json()
    props = data["properties"]["parameter"]

    # NASA POWER mensual devuelve claves tipo "200001", "200002", ...
    # La clave que termina en "13" es el promedio anual — se ignora
    claves = list(props["T2M_MAX"].keys())

    registros = []
    for clave in claves:
        if clave.endswith("13"):
            continue
        ano = int(clave[:4])
        mes = int(clave[4:])
        registros.append({
            "iso_a0":           iso,
            "pais":             nombre,
            "ano":              ano,
            "mes":              mes,
            "tmax_promedio":    props["T2M_MAX"][clave],
            "tmin_promedio":    props["T2M_MIN"][clave],
            "precipitacion":    props["PRECTOTCORR"][clave],
            "humedad_promedio": props["RH2M"][clave],
        })

    df = pd.DataFrame(registros)
    df.replace(-999.0, float("nan"), inplace=True)
    return df


def main():
    total    = len(PAISES)
    todos    = []
    exitosos = []
    fallidos = []

    print("=" * 65)
    print(f"  NASA POWER LATAM — Clima Mensual 2000-2024 ({total} países)")
    print("=" * 65)

    for i, (iso, info) in enumerate(PAISES.items(), 1):
        nombre = info["nombre"]
        lat    = info["lat"]
        lon    = info["lon"]

        print(f"[{i:02d}/{total}] {iso} — {nombre:<40}", end=" ", flush=True)
        try:
            df = descargar_mensual_pais(iso, nombre, lat, lon)
            todos.append(df)
            exitosos.append(iso)
            print(f"OK ({len(df)} meses)")
            time.sleep(1)

        except Exception as e:
            print(f"ERROR: {e}")
            fallidos.append(iso)

    print()
    print("=" * 65)
    print(f"  Exitosos : {len(exitosos)}/{total}")
    if fallidos:
        print(f"  Fallidos : {fallidos}")
    print("=" * 65)

    if not todos:
        print("No se descargó ningún dato.")
        return

    mensual = pd.concat(todos, ignore_index=True)
    mensual = mensual.sort_values(["iso_a0", "ano", "mes"]).reset_index(drop=True)
    mensual.to_csv("nasa_power_latam_mensual.csv", index=False)

    print(f"\nArchivo generado -> nasa_power_latam_mensual.csv")
    print(f"Filas            -> {len(mensual):,}")
    print(f"Paises           -> {mensual['iso_a0'].nunique()}")
    print()
    print("Columnas:")
    print("  iso_a0, pais, ano, mes,")
    print("  tmax_promedio, tmin_promedio,")
    print("  precipitacion (mm mensual acumulado),")
    print("  humedad_promedio")
    print()
    print("Listo para cruzar con OpenDengue en el Agente 2.")


if __name__ == "__main__":
    main()