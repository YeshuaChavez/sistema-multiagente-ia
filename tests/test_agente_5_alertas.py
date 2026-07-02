# -*- coding: utf-8 -*-
"""
Tests de AgenteOrquestador.generar_metricas_finales() (Agente 5).

Fabrica resultados de Agente 3 (xgb_test_lookup) y Agente 4 (lstm_test_lookup)
como si ya hubieran entrenado, sobre un dataset sintetico pequeno, y verifica
que el Agente 5 los combine correctamente: pesos fijos 0.5/0.5, estructura
completa de metrics.json, y los dos bugs de regresion encontrados durante la
auditoria de este proyecto (records_procesados mal calculado, campos de
clasificacion faltantes). No entrena XGBoost ni LSTM reales, y nunca sube
nada a S3 (mockeado via monkeypatch).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import agente_5_alertas as a5

CAMPOS_ESPERADOS = [
    "records_procesados", "n_train", "n_test", "n_paises", "n_departamentos",
    "r2_xgb", "mae_xgb", "rmse_xgb", "r2_lstm", "mae_lstm", "rmse_lstm",
    "r2_ensemble", "mae_ensemble", "rmse_ensemble",
    "ensemble_w_xgb", "ensemble_w_lstm",
    "acc_clasificacion", "kappa_clasificacion",
    "f1_endemico", "f1_alerta", "f1_epidemia",
    "precision_endemico", "precision_alerta", "precision_epidemia",
    "recall_endemico", "recall_alerta", "recall_epidemia",
    "soporte_endemico", "soporte_alerta", "soporte_epidemia",
]


@pytest.fixture(autouse=True)
def no_s3_upload(monkeypatch):
    """Nunca tocar el bucket S3 real durante los tests."""
    monkeypatch.setattr(a5.s3, "upload", lambda local_path, s3_key: None)


@pytest.fixture
def dataset_sintetico(tmp_path):
    """3 departamentos (2 paises) x 4 anios (2019-2022) de incidencia aleatoria."""
    rng = np.random.default_rng(42)
    deptos = [("ARG", "DEPTO_A"), ("ARG", "DEPTO_B"), ("PER", "DEPTO_C")]
    rows = []
    for iso, adm in deptos:
        for ano in [2019, 2020, 2021, 2022]:
            for mes in range(1, 13):
                rows.append({
                    "iso_a0": iso, "pais": iso, "adm_1_name": adm,
                    "ano": ano, "mes": mes,
                    "casos_dengue": 500,  # alto a proposito: pasa el filtro yearly>100
                    "poblacion": 100000,
                    "incidencia_dengue": float(rng.uniform(1, 60)),
                })
    df = pd.DataFrame(rows)

    os.makedirs(tmp_path / "data" / "models", exist_ok=True)
    os.makedirs(tmp_path / "data" / "processed", exist_ok=True)
    feat_path = tmp_path / "data" / "processed" / "dataset_features_latam.csv"
    df.to_csv(feat_path, index=False)
    return str(tmp_path), df, rng


def _fabricar_metricas(df, rng, split_ano=2020):
    """
    Simula xgb_test_lookup / lstm_test_lookup como si vinieran de Agente 3 y
    Agente 4 ya entrenados por separado, con ruido distinto por modelo.
    """
    df_test = df[df["ano"] > split_ano]
    xgb_lookup, lstm_lookup = {}, {}
    for r in df_test.itertuples():
        key = (r.iso_a0.upper(), r.adm_1_name.upper(), int(r.ano), int(r.mes))
        real = r.incidencia_dengue
        xgb_lookup[key] = float(max(0.1, real * rng.uniform(0.85, 1.15)))
        lstm_lookup[key] = float(max(0.1, real * rng.uniform(0.80, 1.20)))

    metricas_ml = {
        "r2_xgb": 0.9149, "mae_xgb": 6.07, "rmse_xgb": 22.18,
        "n_train": int((df["ano"] <= split_ano).sum()), "n_test": len(df_test),
        "xgb_test_lookup": xgb_lookup,
    }
    metricas_dl = {
        "r2_lstm": 0.9035, "mae_lstm": 6.02, "rmse_lstm": 20.52,
        "n_test": len(df_test),
        "lstm_test_lookup": lstm_lookup,
    }
    return metricas_ml, metricas_dl


def test_pesos_del_ensemble_son_fijos_0_5(dataset_sintetico):
    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)

    resultado = a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    assert resultado["ensemble_w_xgb"] == 0.5
    assert resultado["ensemble_w_lstm"] == 0.5


def test_metrics_json_tiene_todos_los_campos_esperados(dataset_sintetico):
    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)

    a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    metrics_path = os.path.join(base_dir, "data", "models", "metrics.json")
    with open(metrics_path) as f:
        escrito = json.load(f)

    faltantes = [c for c in CAMPOS_ESPERADOS if c not in escrito]
    assert not faltantes, f"Faltan campos en metrics.json: {faltantes}"


def test_records_procesados_es_el_dataset_completo_no_solo_train(dataset_sintetico):
    """
    Regresion del bug encontrado en la auditoria: antes 'records_procesados'
    se llenaba con n_train en vez del total del dataset (train + test).
    """
    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)

    resultado = a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    assert resultado["records_procesados"] == len(df)
    assert resultado["records_procesados"] != metricas_ml["n_train"]


def test_n_paises_y_n_departamentos_correctos(dataset_sintetico):
    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)

    resultado = a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    assert resultado["n_paises"] == 2         # ARG, PER
    assert resultado["n_departamentos"] == 3  # DEPTO_A, DEPTO_B, DEPTO_C


def test_ensemble_es_combinacion_honesta_no_solo_xgb_o_solo_lstm(dataset_sintetico):
    """El R² del ensemble debe ser una combinacion real, no filtrarse igual
    al de un solo modelo (que indicaria que el otro no se esta usando)."""
    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)

    resultado = a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    assert resultado["r2_ensemble"] != metricas_ml["r2_xgb"]
    assert resultado["r2_ensemble"] != metricas_dl["r2_lstm"]
    assert 0.0 <= resultado["acc_clasificacion"] <= 1.0
    assert -1.0 <= resultado["kappa_clasificacion"] <= 1.0


def test_no_sube_nada_a_s3_real(dataset_sintetico, monkeypatch):
    """Verifica que la unica subida sea metrics.json, sin tocar otros keys."""
    llamadas = []
    monkeypatch.setattr(a5.s3, "upload", lambda local_path, s3_key: llamadas.append(s3_key))

    base_dir, df, rng = dataset_sintetico
    metricas_ml, metricas_dl = _fabricar_metricas(df, rng)
    a5.AgenteOrquestador.generar_metricas_finales(metricas_ml, metricas_dl, base_dir)

    assert llamadas == ["modelos/metrics.json"]
