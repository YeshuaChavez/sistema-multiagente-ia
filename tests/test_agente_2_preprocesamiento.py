# -*- coding: utf-8 -*-
"""
Tests de AgentePreprocesamiento.generar_features() (Agente 2).

Este es el metodo donde vivia el bug de documentacion "34 vs 73 features"
encontrado durante la auditoria: genera las 73 variables predictoras a
partir del dataset base de 14 columnas (lags climaticos, lags de
incidencia, rolling means, vecinos espaciales, codificacion ciclica,
indicadores binarios, derivadas de clima/incidencia, dummies de pais).

Se prueba generar_features() directamente sobre un df_base sintetico (sin
pasar por procesar_casos/fusionar_datos/calcular_incidencia, que dependen
de las fuentes crudas del Agente 1). El archivo de coordenadas se escribe
localmente en tmp_path para que s3.ensure_local() nunca intente tocar S3
real (ensure_local ya retorna True de inmediato si el archivo local existe).
"""
import os

import numpy as np
import pandas as pd
import pytest

from agente_2_preprocesamiento import AgentePreprocesamiento

# Mismo criterio que usa Agente 3 para separar features de metadata
COLS_EXCLUIR = ['iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
                'casos_dengue', 'poblacion', 'incidencia_dengue']

DEPTOS = [
    ("ARG", "Argentina", "DEPTO_A"),
    ("ARG", "Argentina", "DEPTO_B"),
    ("BOL", "Bolivia", "DEPTO_C"),
    ("BRA", "Brasil", "DEPTO_D"),
    ("COL", "Colombia", "DEPTO_E"),
    ("ECU", "Ecuador", "DEPTO_F"),
    ("MEX", "Mexico", "DEPTO_G"),
    ("PAN", "Panama", "DEPTO_H"),
    ("PER", "Peru", "DEPTO_I"),
]
# Los 8 dummies de pais de produccion (pais_ARG..pais_PER) solo aparecen si
# los 8 paises estan presentes en el dataset -- pd.get_dummies solo genera
# columnas para valores que realmente ocurren en los datos.
N_MESES = 18  # >12 para que sobrevivan filas tras descartar por incidencia_lag12


def _build_df_base():
    rng = np.random.default_rng(7)
    meses = pd.date_range("2020-01-01", periods=N_MESES, freq="MS")
    rows = []
    for iso, pais, adm in DEPTOS:
        for fecha in meses:
            rows.append({
                "iso_a0": iso, "pais": pais, "adm_1_name": adm,
                "ano": fecha.year, "mes": fecha.month,
                "casos_dengue": int(rng.integers(0, 50)),
                "incidencia_dengue": float(rng.uniform(0, 60)),
                "agua_basica": float(rng.uniform(85, 99)),
                "tmax_promedio": float(rng.uniform(20, 35)),
                "tmin_promedio": float(rng.uniform(10, 20)),
                "precipitacion": float(rng.uniform(0, 300)),
                "humedad_promedio": float(rng.uniform(50, 90)),
                "poblacion": 100000,
                "densidad_poblacion": 50.0,
            })
    return pd.DataFrame(rows)


def _write_coords_csv(agente):
    # Coordenadas arbitrarias, una por departamento (ARG tiene 2 para poder
    # probar que se usan entre si como vecinos espaciales dentro del pais).
    coords = pd.DataFrame([
        {"iso_a0": "ARG", "adm_1_name": "DEPTO_A", "lat": -34.0, "lon": -58.0},
        {"iso_a0": "ARG", "adm_1_name": "DEPTO_B", "lat": -35.0, "lon": -59.0},
        {"iso_a0": "BOL", "adm_1_name": "DEPTO_C", "lat": -17.0, "lon": -63.0},
        {"iso_a0": "BRA", "adm_1_name": "DEPTO_D", "lat": -15.0, "lon": -47.0},
        {"iso_a0": "COL", "adm_1_name": "DEPTO_E", "lat": 4.0, "lon": -74.0},
        {"iso_a0": "ECU", "adm_1_name": "DEPTO_F", "lat": -2.0, "lon": -79.0},
        {"iso_a0": "MEX", "adm_1_name": "DEPTO_G", "lat": 23.0, "lon": -102.0},
        {"iso_a0": "PAN", "adm_1_name": "DEPTO_H", "lat": 8.0, "lon": -80.0},
        {"iso_a0": "PER", "adm_1_name": "DEPTO_I", "lat": -9.0, "lon": -75.0},
    ])
    os.makedirs(agente.crudos_dir, exist_ok=True)
    coords.to_csv(os.path.join(agente.crudos_dir, "departamentos_coordenadas.csv"), index=False)


@pytest.fixture
def resultado(tmp_path):
    agente = AgentePreprocesamiento(base_dir=str(tmp_path))
    _write_coords_csv(agente)
    df_base = _build_df_base()
    df_feat = agente.generar_features(df_base)
    return df_base, df_feat


def test_genera_exactamente_73_features(resultado):
    """
    Regresion directa del bug encontrado en la auditoria: el docstring del
    metodo (y el README) decian '34 features' cuando el resto del proyecto
    (paper, metrics.json) siempre confirmo 73.
    """
    _, df_feat = resultado
    cols_feat = [c for c in df_feat.columns if c not in COLS_EXCLUIR]
    assert len(cols_feat) == 73


def test_grupos_de_features_completos(resultado):
    _, df_feat = resultado
    cols = set(df_feat.columns)

    for var in ["tmax", "tmin", "precipitacion", "humedad"]:       # 24 lags climaticos
        for lag in range(1, 7):
            assert f"{var}_lag{lag}" in cols

    for lag in range(1, 13):                                       # 12 lags de incidencia
        assert f"incidencia_lag{lag}" in cols

    assert {"incidencia_roll3", "incidencia_roll6", "incidencia_roll12"} <= cols  # 3 rolling

    for lag in range(1, 7):                                        # 6 vecinos espaciales
        assert f"incidencia_vecinos_lag{lag}" in cols

    assert {"mes_sin", "mes_cos"} <= cols                          # 2 ciclicas
    assert {"indicador_covid", "indicador_nino", "indicador_nina"} <= cols  # 3 binarios

    derivadas = {"amplitud_termica", "temperatura_media", "precipitacion_anomalia",
                 "aceleracion_incidencia", "cambio_interanual", "tendencia_1m",
                 "tendencia_3m", "fase_ascendente", "indicador_brote"}
    assert derivadas <= cols                                        # 9 derivadas

    assert {"pais_ARG", "pais_PER"} <= cols                          # dummies de pais


def test_agua_basica_es_feature_y_poblacion_no(resultado):
    """
    agente_3's COLS_EXCLUIR excluye 'poblacion' de las 73 features (solo es
    metadata usada para calcular incidencia), pero 'agua_basica' si es una
    feature real (el LSTM la usa directamente). Verifica que no se inviertan.
    """
    _, df_feat = resultado
    cols_feat = [c for c in df_feat.columns if c not in COLS_EXCLUIR]
    assert "agua_basica" in cols_feat
    assert "poblacion" not in cols_feat


def test_no_quedan_nan_en_columnas_lag_o_roll(resultado):
    _, df_feat = resultado
    lag_cols = [c for c in df_feat.columns if 'lag' in c or 'roll' in c]
    assert not df_feat[lag_cols].isna().any().any()


def test_descarta_los_primeros_12_meses_por_falta_de_historial(resultado):
    """18 meses de entrada por departamento - 12 perdidos por incidencia_lag12 = 6."""
    _, df_feat = resultado
    assert len(df_feat) == 6 * len(DEPTOS)
    for _, _, adm in DEPTOS:
        assert (df_feat["adm_1_name"] == adm).sum() == 6


def test_tmax_lag1_es_el_tmax_del_mes_anterior(resultado):
    """Detecta bugs de off-by-one en el calculo de lags."""
    df_base, df_feat = resultado
    df_base_a = df_base[df_base["adm_1_name"] == "DEPTO_A"].sort_values(["ano", "mes"]).reset_index(drop=True)
    df_feat_a = df_feat[df_feat["adm_1_name"] == "DEPTO_A"]

    for _, row in df_feat_a.iterrows():
        idx = df_base_a[(df_base_a["ano"] == row["ano"]) & (df_base_a["mes"] == row["mes"])].index[0]
        tmax_mes_anterior = df_base_a.loc[idx - 1, "tmax_promedio"]
        assert row["tmax_lag1"] == pytest.approx(tmax_mes_anterior)


def test_codificacion_ciclica_del_mes_es_correcta(resultado):
    _, df_feat = resultado
    for _, row in df_feat.iterrows():
        mes = row["mes"]
        assert row["mes_sin"] == pytest.approx(np.sin(2 * np.pi * mes / 12))
        assert row["mes_cos"] == pytest.approx(np.cos(2 * np.pi * mes / 12))


def test_dummies_de_pais_son_one_hot(resultado):
    _, df_feat = resultado
    arg_rows = df_feat[df_feat["iso_a0"] == "ARG"]
    per_rows = df_feat[df_feat["iso_a0"] == "PER"]

    assert (arg_rows["pais_ARG"] == 1).all()
    assert (arg_rows["pais_PER"] == 0).all()
    assert (per_rows["pais_PER"] == 1).all()
    assert (per_rows["pais_ARG"] == 0).all()
