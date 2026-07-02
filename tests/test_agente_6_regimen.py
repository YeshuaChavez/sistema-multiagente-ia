# -*- coding: utf-8 -*-
"""
Tests del Agente 6 (Deteccion de Regimen Epidemico).

AgenteRegimen.detectar() es logica pura y deterministica: dado un
lag1/lag2/percentiles fijos, la salida (regimen + pesos ajustados) siempre
es la misma. Cubre los 5 regimenes, sus bordes, y los dos topes distintos
que existen en "Brote activo" (extremidad<=3.0 y w_lstm<=0.80).
"""
import pytest

from agente_6_regimen import AgenteRegimen


@pytest.fixture
def agente():
    """Pesos base de produccion: w_xgb=0.5, w_lstm=0.5."""
    return AgenteRegimen(w_xgb_base=0.5, w_lstm_base=0.5)


def test_normal_debajo_de_p25(agente):
    r = agente.detectar(lag1_raw=5.0, lag1_log=1.79, lag2_log=1.5,
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Normal"
    assert r["w_xgb"] == 0.5 and r["w_lstm"] == 0.5


def test_normal_limite_exacto_en_p25(agente):
    """lag1_raw == p25 debe clasificar como Normal (la condicion usa <=)."""
    r = agente.detectar(lag1_raw=10.0, lag1_log=2.4, lag2_log=2.4,
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Normal"


def test_vigilancia_entre_p25_y_p50(agente):
    r = agente.detectar(lag1_raw=15.0, lag1_log=2.77, lag2_log=2.5,
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Vigilancia"
    assert r["w_xgb"] == 0.5 and r["w_lstm"] == 0.5


def test_vigilancia_en_banda_alta_con_tendencia_decreciente(agente):
    """Entre p50 y p90 pero bajando -> Vigilancia, no Pre-brote."""
    r = agente.detectar(lag1_raw=30.0, lag1_log=2.0, lag2_log=2.5,  # tendencia < 0
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Vigilancia"
    assert r["w_xgb"] == 0.5 and r["w_lstm"] == 0.5


def test_pre_brote_entre_p50_y_p90_con_tendencia_creciente(agente):
    r = agente.detectar(lag1_raw=30.0, lag1_log=2.5, lag2_log=2.0,  # tendencia > 0
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Pre-brote"
    assert r["w_lstm"] == pytest.approx(0.65)  # min(0.5*1.4, 0.65) = 0.65
    assert r["w_xgb"] == pytest.approx(0.35)


def test_brote_activo_sobre_p90_con_tendencia_creciente(agente):
    r = agente.detectar(lag1_raw=60.0, lag1_log=2.5, lag2_log=2.0,  # tendencia > 0
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Brote activo"
    # extremidad = min(60/50, 3.0) = 1.2 -> w_lstm = min(0.5*1.2, 0.80) = 0.6
    assert r["w_lstm"] == pytest.approx(0.6)
    assert r["w_xgb"] == pytest.approx(0.4)


def test_brote_activo_extremidad_topada_en_0_80(agente):
    """Un brote muy severo no debe superar w_lstm=0.80 (tope externo)."""
    r = agente.detectar(lag1_raw=500.0, lag1_log=6.2, lag2_log=5.5,
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Brote activo"
    assert r["w_lstm"] == 0.80
    assert r["w_xgb"] == pytest.approx(0.20)


def test_extremidad_se_topa_en_3_antes_del_tope_de_0_80():
    """
    Aisla el tope interno extremidad=min(ratio, 3.0), distinto del tope
    externo w_lstm<=0.80. Con w_lstm_base=0.15, base*3.0=0.45 (< 0.80),
    asi que si el resultado no fuera 0.45, el tope de 3.0 se habria perdido.
    """
    agente_low_base = AgenteRegimen(w_xgb_base=0.85, w_lstm_base=0.15)
    r = agente_low_base.detectar(lag1_raw=500.0, lag1_log=6.2, lag2_log=5.5,  # ratio = 10x p90
                                  p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Brote activo"
    assert r["w_lstm"] == pytest.approx(0.45)  # min(0.15*3.0, 0.80) = 0.45


def test_post_pico_sobre_p90_con_tendencia_decreciente(agente):
    r = agente.detectar(lag1_raw=60.0, lag1_log=2.0, lag2_log=2.5,  # tendencia < 0
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["regimen"] == "Post-pico"
    assert r["w_xgb"] == pytest.approx(0.75)  # min(0.5*1.5, 0.75) = 0.75
    assert r["w_lstm"] == pytest.approx(0.25)


@pytest.mark.parametrize("lag1_raw,lag1_log,lag2_log", [
    (5.0, 1.79, 1.5),      # Normal
    (15.0, 2.77, 2.5),     # Vigilancia
    (30.0, 2.5, 2.0),      # Pre-brote
    (60.0, 2.5, 2.0),      # Brote activo
    (60.0, 2.0, 2.5),      # Post-pico
])
def test_pesos_siempre_suman_uno(agente, lag1_raw, lag1_log, lag2_log):
    r = agente.detectar(lag1_raw=lag1_raw, lag1_log=lag1_log, lag2_log=lag2_log,
                         p25_local=10.0, p50_local=20.0, p90_local=50.0)
    assert r["w_xgb"] + r["w_lstm"] == pytest.approx(1.0)
