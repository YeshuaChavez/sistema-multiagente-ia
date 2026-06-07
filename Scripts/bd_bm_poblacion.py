"""
AGENTE 1 — Recolección de Datos de Población
Banco Mundial API — países de Latinoamérica y el Caribe

Descarga población total por país/año desde 2000 hasta 2024.
Se usa para calcular la tasa de incidencia (casos / población × 100,000).

Uso:
    pip install requests pandas
    python agente1_banco_mundial_latam.py

Output:
    poblacion_latam_2000_2024.csv → población por país/año lista para cruzar
"""

import requests
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────────────────
# 37 países LATAM — misma lista que el script NASA POWER
# ─────────────────────────────────────────────────────────────────────────────
PAISES_ISO = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
    "HTI", "HND", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY", "VEN",
]

# Territorios sin cobertura en Banco Mundial (todos los 20 están cubiertos)
SIN_DATOS_BM = set()

BASE_URL = "https://api.worldbank.org/v2/country/{iso}/indicator/SP.POP.TOTL"
ANOS     = list(range(2000, 2025))


def descargar_pais(iso):
    """Descarga población total 2000-2024 para un país via Banco Mundial API."""
    resultados = []
    page = 1

    while True:
        url    = BASE_URL.format(iso=iso)
        params = {
            "format":   "json",
            "per_page": 100,
            "page":     page,
            "date":     "2000:2024",
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()

        data = resp.json()

        if len(data) < 2 or not data[1]:
            break

        for registro in data[1]:
            if registro.get("value") is not None:
                resultados.append({
                    "iso_a0":    iso,
                    "ano":       int(registro["date"]),
                    "poblacion": int(registro["value"]),
                })

        total_pages = data[0].get("pages", 1)
        if page >= total_pages:
            break
        page += 1

    return pd.DataFrame(resultados)


def main():
    total    = len(PAISES_ISO) + len(SIN_DATOS_BM)
    todos    = []
    exitosos = []
    sin_datos = list(SIN_DATOS_BM)
    fallidos  = []

    print("=" * 60)
    print(f"  Banco Mundial — Población LATAM 2000-2024")
    print(f"  Con datos: {len(PAISES_ISO)} | Sin datos BM: {len(SIN_DATOS_BM)}")
    print("=" * 60)

    # Notificar territorios sin cobertura
    print(f"\nTerritorios sin cobertura BM (tasa_incidencia = NaN):")
    for iso in sorted(SIN_DATOS_BM):
        print(f"  {iso}")
    print()

    for i, iso in enumerate(PAISES_ISO, 1):
        print(f"[{i:02d}/{len(PAISES_ISO)}] {iso:<5}", end=" ", flush=True)
        try:
            df = descargar_pais(iso)

            if df.empty:
                print(f"SIN DATOS (API vacía)")
                sin_datos.append(iso)
            else:
                todos.append(df)
                exitosos.append(iso)
                print(f"OK ({len(df)} años)")

            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR: {e}")
            fallidos.append(iso)
            time.sleep(2)

    print()
    print("=" * 60)
    print(f"  Con datos    : {len(exitosos)}/{len(PAISES_ISO)}")
    print(f"  Sin datos BM : {len(SIN_DATOS_BM)} territorios")
    if fallidos:
        print(f"  Fallidos     : {fallidos}")
    print("=" * 60)

    if not todos:
        print("No se descargó ningún dato.")
        return

    # ── Construir DataFrame con todos los años completos ──────────────────────
    poblacion = pd.concat(todos, ignore_index=True)

    # Completar años faltantes por país e interpolar
    paises_con_datos = poblacion["iso_a0"].unique()
    anos_completos   = pd.DataFrame(
        [(iso, ano) for iso in paises_con_datos for ano in ANOS],
        columns=["iso_a0", "ano"]
    )
    poblacion = anos_completos.merge(poblacion, on=["iso_a0", "ano"], how="left")
    poblacion = poblacion.sort_values(["iso_a0", "ano"])
    poblacion["poblacion"] = poblacion.groupby("iso_a0")["poblacion"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    poblacion["poblacion"] = poblacion["poblacion"].round(0).astype("Int64")

    poblacion.to_csv("poblacion_latam_2000_2024.csv", index=False)

    print(f"\nArchivo generado -> poblacion_latam_2000_2024.csv")
    print(f"Filas            -> {len(poblacion):,}")
    print(f"Paises con datos -> {poblacion['iso_a0'].nunique()}")
    print()
    print("Columnas:")
    print("  iso_a0    -> codigo ISO (PER, BRA, MEX, etc.)")
    print("  ano       -> ano (2000 a 2024)")
    print("  poblacion -> habitantes totales")
    print()
    print("Listo para cruzar con OpenDengue en el Agente 2.")


if __name__ == "__main__":
    main()