# -*- coding: utf-8 -*-
"""
Descarga el índice ONI (Oceanic Niño Index) desde NOAA y lo guarda como CSV mensual.
ONI mide la anomalía de temperatura superficial del mar en la región Niño 3.4.
  >  +0.5 = El Niño  |  < -0.5 = La Niña  |  entre ambos = Neutro

El índice está fuertemente correlacionado con brotes de dengue en Latinoamérica
con un retardo de 2-4 meses.
"""

import os
import sys
import requests
import pandas as pd

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

NOAA_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

SEASON_TO_MONTH = {
    'DJF': 1,  'JFM': 2,  'FMA': 3,  'MAM': 4,
    'AMJ': 5,  'MJJ': 6,  'JJA': 7,  'JAS': 8,
    'ASO': 9,  'SON': 10, 'OND': 11, 'NDJ': 12,
}


def descargar_oni(output_path):
    print("Descargando indice ONI (ENSO) desde NOAA...")
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()

    records = []
    for line in r.text.splitlines():
        parts = line.split()
        # Formato NOAA: SEAS YR TOTAL ANOM
        if len(parts) < 4:
            continue
        seas = parts[0]
        if seas not in SEASON_TO_MONTH:
            continue
        try:
            yr   = int(parts[1])
            anom = float(parts[3])
        except (ValueError, IndexError):
            continue
        mes = SEASON_TO_MONTH[seas]
        records.append({'ano': yr, 'mes': mes, 'oni': anom})

    df = (
        pd.DataFrame(records)
        .drop_duplicates(['ano', 'mes'])
        .sort_values(['ano', 'mes'])
        .reset_index(drop=True)
    )

    # Calcular lags globales (ONI es global, no por departamento)
    df['oni_lag1'] = df['oni'].shift(1)
    df['oni_lag2'] = df['oni'].shift(2)
    df['oni_lag3'] = df['oni'].shift(3)

    df.to_csv(output_path, index=False)
    print(f"   -> {len(df)} registros ONI guardados en {output_path}")
    print(f"   -> Rango: {df['ano'].min()}-{df['ano'].max()}")
    return df


if __name__ == "__main__":
    base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
    out = os.path.join(base_dir, "Base de Datos", "datos_crudos", "oni_mensual.csv")
    descargar_oni(out)
