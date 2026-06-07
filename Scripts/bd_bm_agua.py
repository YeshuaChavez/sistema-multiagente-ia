"""
AGENTE 1 — Recolección de Datos de Acceso al Agua
Banco Mundial API — 20 países de Latinoamérica

Descarga:
  1. SH.H2O.BASW.ZS -> % población con acceso a agua básica (al menos)
  2. SH.H2O.SMDW.ZS -> % población con acceso a agua gestionada de forma segura

Uso:
    python Scripts/bd_bm_agua.py
"""

import requests
import pandas as pd
import time

PAISES_ISO = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
    "HTI", "HND", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY", "VEN",
]

INDICADORES = {
    "agua_basica": "SH.H2O.BASW.ZS",
    "agua_segura": "SH.H2O.SMDW.ZS"
}

BASE_URL = "https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"
ANOS     = list(range(2000, 2025))

def descargar_indicador(iso, indicator_name, indicator_code):
    """Descarga el indicador para un país y rango de años."""
    resultados = []
    page = 1
    
    while True:
        url = BASE_URL.format(iso=iso, indicator=indicator_code)
        params = {
            "format": "json",
            "per_page": 100,
            "page": page,
            "date": "2000:2024"
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        
        data = resp.json()
        
        if len(data) < 2 or not data[1]:
            break
            
        for registro in data[1]:
            val = registro.get("value")
            resultados.append({
                "iso_a0": iso,
                "ano": int(registro["date"]),
                indicator_name: val if val is not None else float("nan")
            })
            
        total_pages = data[0].get("pages", 1)
        if page >= total_pages:
            break
        page += 1
        
    return pd.DataFrame(resultados)

def main():
    print("=" * 60)
    print("  Banco Mundial — Acceso al Agua LATAM (20 Paises)")
    print("=" * 60)
    
    todos_datos = []
    
    for i, iso in enumerate(PAISES_ISO, 1):
        print(f"[{i:02d}/20] Descargando {iso}...", end=" ", flush=True)
        try:
            # Descargar agua básica
            df_basica = descargar_indicador(iso, "agua_basica", INDICADORES["agua_basica"])
            # Descargar agua segura
            df_segura = descargar_indicador(iso, "agua_segura", INDICADORES["agua_segura"])
            
            if df_basica.empty and df_segura.empty:
                print("SIN DATOS")
                continue
                
            # Combinar ambos indicadores
            if not df_basica.empty and not df_segura.empty:
                df_pais = pd.merge(df_basica, df_segura, on=["iso_a0", "ano"], how="outer")
            elif not df_basica.empty:
                df_pais = df_basica
                df_pais["agua_segura"] = float("nan")
            else:
                df_pais = df_segura
                df_pais["agua_basica"] = float("nan")
                
            todos_datos.append(df_pais)
            print("OK")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(1)
            
    if not todos_datos:
        print("No se descargo ningun dato.")
        return
        
    df_final = pd.concat(todos_datos, ignore_index=True)
    
    # Rellenar años faltantes por país e interpolar para suavizar la serie
    paises_con_datos = df_final["iso_a0"].unique()
    anos_completos = pd.DataFrame(
        [(iso, ano) for iso in paises_con_datos for ano in ANOS],
        columns=["iso_a0", "ano"]
    )
    df_final = pd.merge(anos_completos, df_final, on=["iso_a0", "ano"], how="left")
    df_final = df_final.sort_values(["iso_a0", "ano"])
    
    # Interpolar linealmente valores nulos dentro de cada país para completar huecos intermedios
    df_final["agua_basica"] = df_final.groupby("iso_a0")["agua_basica"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    df_final["agua_segura"] = df_final.groupby("iso_a0")["agua_segura"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    
    # Redondear a 2 decimales
    df_final["agua_basica"] = df_final["agua_basica"].round(2)
    df_final["agua_segura"] = df_final["agua_segura"].round(2)
    
    output_csv = "Scripts/agua_latam_2000_2024.csv"
    df_final.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"Archivo generado -> {output_csv}")
    print(f"Filas            -> {len(df_final):,}")
    print(f"Países           -> {df_final['iso_a0'].nunique()}")
    print("=" * 60)
    
    # Mostrar resumen de completitud y valores promedio por país
    resumen = df_final.groupby("iso_a0").agg(
        basica_promedio=("agua_basica", "mean"),
        basica_nulos=("agua_basica", lambda x: x.isnull().sum()),
        segura_promedio=("agua_segura", "mean"),
        segura_nulos=("agua_segura", lambda x: x.isnull().sum())
    )
    print("\nResumen de cobertura e indices promedio por pais:")
    print(resumen.to_string())

if __name__ == "__main__":
    main()
