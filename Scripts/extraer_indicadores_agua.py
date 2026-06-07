import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("  Extracción e Imputación de Indicadores de Agua Detallados (JMP)")
    print("=" * 60)
    
    file_path = "Scripts/jmp_global_household.xlsx"
    print(f"Leyendo Excel de JMP: {file_path}...")
    
    # Leer hoja 'wat'
    df = pd.read_excel(file_path, sheet_name='wat')
    
    # 20 Países Latinoamericanos seleccionados
    PAISES_ISO = [
        "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
        "HTI", "HND", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY", "VEN"
    ]
    
    # Filtrar por los 20 países y el periodo de interés (2000-2024)
    df_latam = df[(df['iso3'].isin(PAISES_ISO)) & (df['year'] >= 2000) & (df['year'] <= 2024)].copy()
    
    # Columnas JMP de interés y su mapeo
    col_mapping = {
        'wat_basal_t': 'agua_basica',
        'wat_lim_t': 'agua_limitada',
        'wat_unimp_t': 'agua_no_mejorada',
        'wat_ns_t': 'agua_superficial',
        'wat_pip_t': 'agua_entubada',
        'wat_imp_npip_t': 'agua_no_entubada',
        'wat_imp_av_t': 'agua_disponible',
        'wat_sm_t': 'agua_segura'
    }
    
    # Mantener solo las columnas necesarias
    cols_to_keep = ['iso3', 'year'] + list(col_mapping.keys())
    df_latam = df_latam[cols_to_keep].copy()
    df_latam.rename(columns={'iso3': 'iso_a0', 'year': 'ano'}, inplace=True)
    
    # Nicaragua: Datos oficiales del Portal SDG 6 para agua básica y segura
    nicaragua_basica = {
        2000: 81.25, 2001: 81.34, 2002: 81.43, 2003: 81.51, 2004: 81.60,
        2005: 81.69, 2006: 81.79, 2007: 81.90, 2008: 82.01, 2009: 82.11,
        2010: 82.22, 2011: 82.33, 2012: 82.44, 2013: 82.55, 2014: 82.67,
        2015: 82.78, 2016: 82.89, 2017: 82.97, 2018: 83.04, 2019: 83.13,
        2020: 83.22, 2021: 83.22, 2022: 83.22, 2023: 83.22, 2024: 83.22,
    }

    nicaragua_segura = {
        2000: 47.85, 2001: 47.95, 2002: 48.05, 2003: 48.15, 2004: 48.25,
        2005: 48.36, 2006: 48.48, 2007: 48.60, 2008: 49.64, 2009: 50.66,
        2010: 51.67, 2011: 52.67, 2012: 53.65, 2013: 54.63, 2014: 55.08,
        2015: 55.17, 2016: 55.26, 2017: 55.31, 2018: 55.37, 2019: 55.44,
        2020: 55.52, 2021: 55.52, 2022: 55.52, 2023: 55.52, 2024: 55.52,
    }
    
    # Vecinos de Nicaragua en Centroamérica para imputación proporcional
    VECINOS_NIC = ["HND", "SLV", "GTM"]
    
    print("Realizando la imputación proporcional para Nicaragua (NIC) usando vecinos de CA...")
    
    # Iterar año por año para calcular las proporciones de los vecinos e imputar a Nicaragua
    for year in range(2000, 2025):
        # Filtrar datos de vecinos para este año
        df_vecinos_y = df_latam[(df_latam['ano'] == year) & (df_latam['iso_a0'].isin(VECINOS_NIC))]
        
        props_lim = []
        props_unimp = []
        props_ns = []
        props_pip = []
        props_av = []
        
        for _, row in df_vecinos_y.iterrows():
            w_basal = row['wat_basal_t']
            w_lim = row['wat_lim_t']
            w_unimp = row['wat_unimp_t']
            w_ns = row['wat_ns_t']
            w_pip = row['wat_pip_t']
            w_av = row['wat_imp_av_t']
            
            # Proporción de Limited, Unimproved, Surface en la población NO básica
            non_basic = w_lim + w_unimp + w_ns if pd.notnull(w_lim) and pd.notnull(w_unimp) and pd.notnull(w_ns) else np.nan
            if pd.notnull(non_basic) and non_basic > 0:
                props_lim.append(w_lim / non_basic)
                props_unimp.append(w_unimp / non_basic)
                props_ns.append(w_ns / non_basic)
                
            # Proporción de Piped en el agua mejorada (Basic + Limited)
            improved = w_basal + w_lim if pd.notnull(w_basal) and pd.notnull(w_lim) else np.nan
            if pd.notnull(improved) and improved > 0 and pd.notnull(w_pip):
                props_pip.append(w_pip / improved)
                
            # Proporción de Disponibilidad continua en el acceso básico
            if pd.notnull(w_basal) and w_basal > 0 and pd.notnull(w_av):
                props_av.append(w_av / w_basal)
                
        # Promediar proporciones de los vecinos
        p_lim_avg = np.mean(props_lim) if props_lim else 0.33
        p_unimp_avg = np.mean(props_unimp) if props_unimp else 0.33
        p_ns_avg = np.mean(props_ns) if props_ns else 0.33
        
        # Normalizar para que sumen 1
        total_p = p_lim_avg + p_unimp_avg + p_ns_avg
        if total_p > 0:
            p_lim_avg /= total_p
            p_unimp_avg /= total_p
            p_ns_avg /= total_p
        else:
            p_lim_avg, p_unimp_avg, p_ns_avg = 1/3, 1/3, 1/3
            
        p_pip_avg = np.mean(props_pip) if props_pip else 0.85
        p_av_avg = np.mean(props_av) if props_av else 0.70
        
        # Asignar a Nicaragua
        nic_basal = nicaragua_basica[year]
        nic_segura = nicaragua_segura[year]
        rem_nic = 100.0 - nic_basal
        
        nic_lim = p_lim_avg * rem_nic
        nic_unimp = p_unimp_avg * rem_nic
        nic_ns = p_ns_avg * rem_nic
        
        nic_improved = nic_basal + nic_lim
        nic_pip = p_pip_avg * nic_improved
        nic_npip = nic_improved - nic_pip
        nic_av = p_av_avg * nic_basal
        
        # Guardar en el DataFrame original para Nicaragua
        mask = (df_latam['iso_a0'] == "NIC") & (df_latam['ano'] == year)
        df_latam.loc[mask, 'wat_basal_t'] = nic_basal
        df_latam.loc[mask, 'wat_lim_t'] = nic_lim
        df_latam.loc[mask, 'wat_unimp_t'] = nic_unimp
        df_latam.loc[mask, 'wat_ns_t'] = nic_ns
        df_latam.loc[mask, 'wat_pip_t'] = nic_pip
        df_latam.loc[mask, 'wat_imp_npip_t'] = nic_npip
        df_latam.loc[mask, 'wat_imp_av_t'] = nic_av
        df_latam.loc[mask, 'wat_sm_t'] = nic_segura

    print("Imputación de Nicaragua completada.")

    # Interpolación y Extrapolación lineal para rellenar vacíos intermedios/extremos por país
    print("Aplicando interpolación y extrapolación lineal por país...")
    for col in col_mapping.keys():
        df_latam[col] = df_latam.groupby("iso_a0")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    # Imputación Socioeconómica Regional para variables que siguen siendo 100% nulas en algunos países
    # Por ejemplo, agua_segura (wat_sm_t) y agua_disponible (wat_imp_av_t) no tienen ningún dato histórico en algunos países.
    print("Aplicando imputación regional proporcional para países con datos 100% nulos...")
    
    for year in range(2000, 2025):
        df_y = df_latam[df_latam['ano'] == year]
        
        # 1. Ratio para agua segura: wat_sm_t / wat_basal_t
        valid_sm = df_y[df_y['wat_sm_t'].notnull() & df_y['wat_basal_t'].notnull()]
        ratio_sm = (valid_sm['wat_sm_t'] / valid_sm['wat_basal_t']).mean() if len(valid_sm) > 0 else 0.65
        
        # 2. Ratio para agua disponible: wat_imp_av_t / wat_basal_t
        valid_av = df_y[df_y['wat_imp_av_t'].notnull() & df_y['wat_basal_t'].notnull()]
        ratio_av = (valid_av['wat_imp_av_t'] / valid_av['wat_basal_t']).mean() if len(valid_av) > 0 else 0.85
        
        # Aplicar imputación si es nulo
        mask_sm_null = df_latam['wat_sm_t'].isnull() & (df_latam['ano'] == year)
        df_latam.loc[mask_sm_null, 'wat_sm_t'] = df_latam.loc[mask_sm_null, 'wat_basal_t'] * ratio_sm
        
        mask_av_null = df_latam['wat_imp_av_t'].isnull() & (df_latam['ano'] == year)
        df_latam.loc[mask_av_null, 'wat_imp_av_t'] = df_latam.loc[mask_av_null, 'wat_basal_t'] * ratio_av

    # Asegurar que todas las columnas de agua estén acotadas entre 0 y 100
    for col in col_mapping.keys():
        df_latam[col] = df_latam[col].clip(0.0, 100.0)
        
    # Renombrar columnas
    df_latam.rename(columns=col_mapping, inplace=True)
    
    # Ordenar y guardar
    df_latam = df_latam.sort_values(['iso_a0', 'ano']).reset_index(drop=True)
    
    # Redondear indicadores a 4 decimales para alta precisión
    indicator_cols = list(col_mapping.values())
    df_latam[indicator_cols] = df_latam[indicator_cols].round(4)
    
    output_path = "Scripts/agua_latam_detallado.csv"
    df_latam.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"Archivo generado -> {output_path}")
    print(f"Filas            -> {len(df_latam)}")
    print(f"Países           -> {df_latam['iso_a0'].nunique()}")
    print(f"Valores nulos    -> {df_latam.isnull().sum().sum()}")
    print("=" * 60)
    
    # Mostrar resumen
    print("\nPrimeras 5 filas del archivo generado:")
    print(df_latam.head())
    
    print("\nResumen descriptivo de los indicadores:")
    print(df_latam[indicator_cols].describe().to_string())

if __name__ == "__main__":
    main()
