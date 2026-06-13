# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 2: Preprocesamiento y Feature Engineering
--------------------------------------------------
Responsabilidad: Limpieza, transformación, fusión de fuentes multidominio,
cálculo de tasas de incidencia, y generación de todas las variables predictoras
(lags temporales, rolling means, vecinos espaciales, codificación cíclica).
Produce dos artefactos en S3:
  - dataset_maestro_mensual_latam.csv  (14 columnas base, para el backend)
  - dataset_features_latam.csv         (34 features, para entrenamiento de Agentes 3 y 4)
"""

import os
import sys
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import s3_client as s3


class AgentePreprocesamiento:
    def __init__(self, base_dir=None):
        if base_dir is None:
            if os.path.exists("/app"):
                self.base_dir = "/app"
            else:
                self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.db_dir        = os.path.join(self.base_dir, "Base de Datos")
        self.crudos_dir    = os.path.join(self.db_dir, "datos_crudos")
        self.procesados_dir = os.path.join(self.db_dir, "datos_procesados")
        self.output_base   = os.path.join(self.procesados_dir, "dataset_maestro_mensual_latam.csv")
        self.output_feat   = os.path.join(self.procesados_dir, "dataset_features_latam.csv")

    # ─────────────────────────────────────────────────────────────
    # PASO 1 — Procesar casos de dengue
    # ─────────────────────────────────────────────────────────────

    def procesar_casos(self, df_casos):
        print("[Agente 2] Procesando casos epidemiológicos a nivel mensual...")
        df = df_casos.copy()
        df['iso_a0'] = df['ISO_A0']
        df['date']   = pd.to_datetime(df['calendar_start_date'])
        df['ano']    = df['date'].dt.year
        df['mes']    = df['date'].dt.month
        df_m = df.groupby(['iso_a0', 'adm_1_name', 'ano', 'mes'], as_index=False)['dengue_total'].sum()
        df_m.rename(columns={'dengue_total': 'casos_dengue'}, inplace=True)
        print(f"   [Casos] Shape: {df_m.shape}")
        return df_m

    # ─────────────────────────────────────────────────────────────
    # PASO 2 — Fusionar fuentes
    # ─────────────────────────────────────────────────────────────

    def fusionar_datos(self, df_casos_m, df_clima, df_agua, df_poblacion):
        print("[Agente 2] Fusionando fuentes de datos...")
        df_casos_m  = df_casos_m.copy()
        df_clima    = df_clima.copy()
        df_agua     = df_agua.copy()
        df_poblacion = df_poblacion.copy()

        for df in [df_casos_m, df_clima, df_agua]:
            df['ano']       = df['ano'].astype(int)
            df['mes']       = df['mes'].astype(int)
            df['iso_a0']    = df['iso_a0'].astype(str).str.strip().str.upper()
            df['adm_1_name'] = df['adm_1_name'].astype(str).str.strip()

        df_poblacion['ano']       = df_poblacion['ano'].astype(int)
        df_poblacion['iso_a0']    = df_poblacion['iso_a0'].astype(str).str.strip().str.upper()
        df_poblacion['adm_1_name'] = df_poblacion['adm_1_name'].astype(str).str.strip()

        df_m = pd.merge(df_clima, df_casos_m, on=['iso_a0', 'adm_1_name', 'ano', 'mes'], how='left')
        df_m['casos_dengue'] = df_m['casos_dengue'].fillna(0).astype(int)

        df_m = pd.merge(df_m, df_agua[['iso_a0', 'adm_1_name', 'ano', 'mes', 'agua_basica']],
                        on=['iso_a0', 'adm_1_name', 'ano', 'mes'], how='left')
        df_m['agua_basica'] = df_m.groupby(['iso_a0', 'adm_1_name'])['agua_basica'].transform(
            lambda x: x.ffill().bfill()
        )
        df_m['agua_basica'] = df_m.groupby(['iso_a0'])['agua_basica'].transform(
            lambda x: x.fillna(x.median())
        )

        df_m = pd.merge(df_m, df_poblacion[['iso_a0', 'adm_1_name', 'ano', 'poblacion']],
                        on=['iso_a0', 'adm_1_name', 'ano'], how='inner')

        print(f"   [Fusión] {df_m.shape[0]} registros integrados.")
        return df_m

    # ─────────────────────────────────────────────────────────────
    # PASO 3 — Calcular incidencia y densidad
    # ─────────────────────────────────────────────────────────────

    def calcular_incidencia(self, df_merged):
        print("[Agente 2] Calculando incidencia y densidad poblacional...")
        df = df_merged.copy()
        df['incidencia_dengue'] = (df['casos_dengue'] / df['poblacion'] * 100000).round(4)

        areas_path = os.path.join(self.crudos_dir, "departamentos_areas.csv")
        s3.ensure_local(s3.PREFIX_CRUDOS + "departamentos_areas.csv", areas_path)
        if os.path.exists(areas_path):
            df_areas = pd.read_csv(areas_path)
            df_areas['iso_a0']    = df_areas['iso_a0'].astype(str).str.strip().str.upper()
            df_areas['adm_1_name'] = df_areas['adm_1_name'].astype(str).str.strip()
            df = pd.merge(df, df_areas[['iso_a0', 'adm_1_name', 'area_km2']],
                          on=['iso_a0', 'adm_1_name'], how='left')
            df['densidad_poblacion'] = (df['poblacion'] / df['area_km2']).round(4)
            df['densidad_poblacion'] = df['densidad_poblacion'].fillna(df['densidad_poblacion'].median())
        else:
            df['densidad_poblacion'] = 0.0

        df = df.sort_values(['pais', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        base_cols = [
            'iso_a0', 'pais', 'adm_1_name', 'ano', 'mes', 'casos_dengue',
            'incidencia_dengue', 'agua_basica', 'tmax_promedio', 'tmin_promedio',
            'precipitacion', 'humedad_promedio', 'poblacion', 'densidad_poblacion'
        ]
        df = df[base_cols]
        print(f"   [Incidencia] Dataset base: {df.shape[0]} registros, {df.shape[1]} columnas.")
        return df

    # ─────────────────────────────────────────────────────────────
    # PASO 4 — Feature Engineering (responsabilidad del Agente 2)
    # ─────────────────────────────────────────────────────────────

    def generar_features(self, df_base):
        """
        Genera las 34 variables predictoras a partir del dataset base de 14 columnas:
          - Lags climáticos 1-3 (tmax, tmin, precipitación, humedad)
          - Lags de incidencia 1-6
          - Rolling means de incidencia (ventanas 3 y 6 meses)
          - Vecinos espaciales: incidencia media de los 3 departamentos más cercanos, lags 1-6
          - Codificación cíclica del mes (sin/cos)
        """
        print("[Agente 2] Generando features predictoras (lags, rolling, vecinos, estacionalidad)...")
        df = df_base.copy()
        df = df.sort_values(['iso_a0', 'adm_1_name', 'ano', 'mes']).reset_index(drop=True)
        grp = df.groupby(['iso_a0', 'adm_1_name'])

        # Lags climáticos 1-3
        for var in ['tmax_promedio', 'tmin_promedio', 'precipitacion', 'humedad_promedio']:
            base = var.split('_')[0] if 'promedio' in var else var
            for lag in [1, 2, 3]:
                df[f"{base}_lag{lag}"] = grp[var].shift(lag)

        # Lags de incidencia 1-6
        for lag in range(1, 7):
            df[f"incidencia_lag{lag}"] = grp['incidencia_dengue'].shift(lag)

        # Rolling means (shift(1) para evitar data leakage)
        df['incidencia_roll3'] = grp['incidencia_dengue'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df['incidencia_roll6'] = grp['incidencia_dengue'].transform(
            lambda x: x.shift(1).rolling(6, min_periods=1).mean()
        )

        # Codificación cíclica del mes
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        # Vecinos espaciales (3 departamentos más cercanos por país) con lags 1-6
        coords_path = os.path.join(self.crudos_dir, "departamentos_coordenadas.csv")
        s3.ensure_local(s3.PREFIX_CRUDOS + "departamentos_coordenadas.csv", coords_path)
        if os.path.exists(coords_path):
            df_coords = pd.read_csv(coords_path)
            df_coords['iso_a0']    = df_coords['iso_a0'].astype(str).str.strip().str.upper()
            df_coords['adm_1_name'] = df_coords['adm_1_name'].astype(str).str.strip().str.upper()

            df['adm_upper'] = df['adm_1_name'].astype(str).str.strip().str.upper()

            neighbors = {}
            for country in df_coords['iso_a0'].unique():
                cc = df_coords[df_coords['iso_a0'] == country]
                depts  = cc['adm_1_name'].values
                coords = cc[['lat', 'lon']].values
                for i, d_i in enumerate(depts):
                    dists = sorted(
                        [(depts[j], np.sqrt((coords[i, 0] - coords[j, 0])**2 +
                                            (coords[i, 1] - coords[j, 1])**2))
                         for j in range(len(depts)) if j != i],
                        key=lambda x: x[1]
                    )
                    neighbors[(country, d_i)] = [d[0] for d in dists[:3]]

            lookup = {(r.iso_a0, r.adm_upper, r.ano, r.mes): r.incidencia_dengue
                      for r in df.itertuples()}

            inc_vec = []
            for row in df.itertuples():
                nbrs = neighbors.get((row.iso_a0, row.adm_upper), [])
                vals = [lookup.get((row.iso_a0, n, row.ano, row.mes)) for n in nbrs]
                vals = [v for v in vals if v is not None]
                inc_vec.append(np.mean(vals) if vals else row.incidencia_dengue)

            df['incidencia_vecinos'] = inc_vec
            grp_u = df.groupby(['iso_a0', 'adm_upper'])
            for lag in range(1, 7):
                df[f'incidencia_vecinos_lag{lag}'] = grp_u['incidencia_vecinos'].shift(lag)

            df.drop(columns=['adm_upper', 'incidencia_vecinos'], inplace=True)
        else:
            print("   [Advertencia] Sin coordenadas — vecinos en 0.")
            for lag in range(1, 7):
                df[f'incidencia_vecinos_lag{lag}'] = 0.0

        # Eliminar filas con NaN introducidos por lags
        lag_cols = [c for c in df.columns if 'lag' in c or 'roll' in c]
        df.dropna(subset=lag_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"   [Features] Dataset features: {df.shape[0]} registros, {df.shape[1]} columnas.")
        return df

    # ─────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    def ejecutar_preprocesamiento(self, datos_crudos):
        print("=" * 70)
        print("  EJECUTANDO PREPROCESAMIENTO — AGENTE 2")
        print("=" * 70)

        os.makedirs(self.procesados_dir, exist_ok=True)

        df_casos_m = self.procesar_casos(datos_crudos['casos'])
        df_merged  = self.fusionar_datos(
            df_casos_m, datos_crudos['clima'],
            datos_crudos['agua'], datos_crudos['poblacion']
        )
        df_base = self.calcular_incidencia(df_merged)
        df_base = df_base[df_base['ano'] <= 2022].reset_index(drop=True)

        # Guardar dataset base (14 cols) → backend
        df_base.to_csv(self.output_base, index=False)
        s3.upload(self.output_base, s3.PREFIX_PROCESADOS + "dataset_maestro_mensual_latam.csv")
        print(f"   [S3] Dataset base subido.")

        # Feature engineering → entrenamiento Agentes 3 y 4
        df_feat = self.generar_features(df_base)
        df_feat.to_csv(self.output_feat, index=False)
        s3.upload(self.output_feat, s3.PREFIX_PROCESADOS + "dataset_features_latam.csv")
        print(f"   [S3] Dataset features subido.")

        print("SUCCESS: [Agente 2] Preprocesamiento y feature engineering completados.")
        print("=" * 70)
        return df_base


if __name__ == "__main__":
    from agente_1_recoleccion import AgenteRecoleccion
    datos = AgenteRecoleccion().ejecutar_ingesta()
    AgentePreprocesamiento().ejecutar_preprocesamiento(datos)
