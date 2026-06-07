import pandas as pd
import os

# List of 20 Latin American countries requested by the user
PAISES_20 = {
    "ARG": "ARGENTINA",
    "BOL": "BOLIVIA",
    "BRA": "BRAZIL",
    "CHL": "CHILE",
    "COL": "COLOMBIA",
    "CRI": "COSTA RICA",
    "CUB": "CUBA",
    "DOM": "DOMINICAN REPUBLIC",
    "ECU": "ECUADOR",
    "GTM": "GUATEMALA",
    "HTI": "HAITI",
    "HND": "HONDURAS",
    "MEX": "MEXICO",
    "NIC": "NICARAGUA",
    "PAN": "PANAMA",
    "PER": "PERU",
    "PRY": "PARAGUAY",
    "SLV": "EL SALVADOR",
    "URY": "URUGUAY",
    "VEN": "VENEZUELA"
}

def main():
    dengue_path = r"Base de Datos\National_extract_V1_3.csv"
    output_path = r"Base de Datos\dengue_mensual_casos.csv"
    
    print("=" * 60)
    print("  Agregacion de Dengue Mensual Real (20 Paises)")
    print("=" * 60)
    
    if not os.path.exists(dengue_path):
        print(f"Error: No se encontro el archivo {dengue_path}")
        return
        
    print(f"Cargando {dengue_path}...")
    df = pd.read_csv(dengue_path, low_memory=False)
    
    # 1. Filtrar por los 20 países de interés
    df = df[df["ISO_A0"].isin(PAISES_20.keys())].copy()
    print(f"Filtrado para 20 paises: {len(df)} registros")
    
    # 2. Filtrar solo resoluciones temporales Week y Month (descartar Year para evitar simulaciones)
    df = df[df["T_res"].isin(["Week", "Month"])].copy()
    print(f"Filtrado a resoluciones 'Week' y 'Month': {len(df)} registros")
    
    if df.empty:
        print("Error: No quedaron registros despues del filtrado temporal.")
        return
        
    # 3. Parsear fechas y extraer mes/año
    # calendar_start_date se usa como referencia temporal
    df["start_date"] = pd.to_datetime(df["calendar_start_date"], errors='coerce')
    df = df.dropna(subset=["start_date"]).copy()
    
    df["ano"] = df["start_date"].dt.year
    df["mes"] = df["start_date"].dt.month
    
    # Filtrar al período de interés (2000 a 2024)
    df = df[(df["ano"] >= 2000) & (df["ano"] <= 2024)].copy()
    print(f"Filtrado para periodo 2000-2024: {len(df)} registros")
    
    # 4. Agrupar y sumar casos reales por país, año y mes
    # Agrupamos por ISO_A0, ano, mes
    df_mensual = df.groupby(["ISO_A0", "ano", "mes"], as_index=False).agg(
        casos=("dengue_total", "sum")
    )
    
    # Renombrar columnas para estandarizar
    df_mensual.rename(columns={"ISO_A0": "iso_a0"}, inplace=True)
    
    # Agregar el nombre legible del país
    df_mensual["pais"] = df_mensual["iso_a0"].map(PAISES_20)
    
    # Reordenar columnas
    df_mensual = df_mensual[["iso_a0", "pais", "ano", "mes", "casos"]]
    df_mensual = df_mensual.sort_values(["iso_a0", "ano", "mes"]).reset_index(drop=True)
    
    # Guardar en CSV
    df_mensual.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"Archivo generado  -> {output_path}")
    print(f"Filas resultantes -> {len(df_mensual):,}")
    print(f"Paises incluidos  -> {df_mensual['iso_a0'].nunique()}")
    print("=" * 60)
    
    # Mostrar resumen de datos por país
    resumen = df_mensual.groupby("pais").agg(
        meses_totales=("mes", "count"),
        total_casos=("casos", "sum"),
        ano_inicio=("ano", "min"),
        ano_fin=("ano", "max")
    ).sort_values("total_casos", ascending=False)
    
    print("\nResumen por Pais:")
    print(resumen.to_string())
    print("\nTodo listo. Datos agregados y guardados sin simulacion.")

if __name__ == "__main__":
    main()
