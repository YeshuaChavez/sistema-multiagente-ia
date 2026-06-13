# -*- coding: utf-8 -*-
"""
SMA-ML/DL — Backend Services
-----------------------------
Capa de servicio REST que inicializa el Sistema Multi-Agente y delega
toda la lógica de predicción a los Agentes 3, 4 y 5.
"""

import os
import sys
import importlib.util
import pandas as pd


def _load_agente(module_name: str, rel_path: str):
    """Carga un módulo de agente por ruta de archivo (independiente de sys.path)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(root, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_s3 = _load_agente("s3_client",             "agentes/s3_client.py")
_a3 = _load_agente("agente_3_prediccion_ml", "agentes/agente_3_prediccion_ml.py")
_a4 = _load_agente("agente_4_prediccion_dl", "agentes/agente_4_prediccion_dl.py")
_a5 = _load_agente("agente_5_alertas",        "agentes/agente_5_alertas.py")

AgentePrediccionML = _a3.AgentePrediccionML
AgentePrediccionDL = _a4.AgentePrediccionDL
AgenteOrquestador  = _a5.AgenteOrquestador


class PredictionService:
    """
    Servicio de predicción REST.
    Inicializa el Sistema Multi-Agente (Agentes 3, 4 y 5) y expone
    sus capacidades como métodos para los endpoints FastAPI.
    """

    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
        else:
            self.base_dir = base_dir

        self.db_dir        = os.path.join(self.base_dir, "Base de Datos")
        self.model_dir     = os.path.join(self.db_dir, "modelos")
        self.processed_dir = os.path.join(self.db_dir, "datos_procesados")
        self.raw_dir       = os.path.join(self.db_dir, "datos_crudos")

        self.orquestador: AgenteOrquestador = None
        self.inicializar_servicio()

    # ─── Propiedades de acceso directo (compatibilidad con endpoints FastAPI) ───

    @property
    def df_master(self):
        return self.orquestador.df_master if self.orquestador else None

    @property
    def df_coords(self):
        return self.orquestador.df_coords if self.orquestador else None

    @property
    def cols_feat(self):
        return self.orquestador.agente_ml.cols_feat if self.orquestador else None

    @property
    def p25(self):
        return self.orquestador.p25 if self.orquestador else 0.0

    @property
    def p50(self):
        return self.orquestador.p50 if self.orquestador else 0.0

    @property
    def p90(self):
        return self.orquestador.p90 if self.orquestador else 0.0

    @property
    def shap_importance(self):
        return self.orquestador.agente_ml.shap_importance if self.orquestador else None

    # ─── Inicialización del Sistema Multi-Agente ───

    def _descargar_desde_s3(self):
        """Descarga modelos y datos desde S3 si no existen localmente."""
        print("[SMA-ML/DL] Verificando archivos en S3...")
        modelos = [
            "lgbm_model.pkl", "imputador_ml.pkl", "escalador_ml.pkl",
            "cols_feat.pkl", "shap_importance.json",
            "lstm_model.pth", "lstm_config.json", "lstm_features.pkl",
            "escalador_lstm.pkl", "metrics.json",
        ]
        for fname in modelos:
            local = os.path.join(self.model_dir, fname)
            _s3.ensure_local(_s3.PREFIX_MODELOS + fname, local)

        _s3.ensure_local(
            _s3.PREFIX_PROCESADOS + "dataset_maestro_mensual_latam.csv",
            os.path.join(self.processed_dir, "dataset_maestro_mensual_latam.csv")
        )
        _s3.ensure_local(
            _s3.PREFIX_CRUDOS + "departamentos_coordenadas.csv",
            os.path.join(self.raw_dir, "departamentos_coordenadas.csv")
        )

    def inicializar_servicio(self):
        print("[SMA-ML/DL] Iniciando Sistema Multi-Agente...")
        self._descargar_desde_s3()

        # Dataset maestro
        master_path = os.path.join(self.processed_dir, "dataset_maestro_mensual_latam.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Falta dataset maestro: {master_path}")
        df_master = pd.read_csv(master_path)
        print(f"   -> Dataset maestro: {df_master.shape[0]} registros.")

        # Coordenadas departamentales (opcional)
        df_coords = None
        coords_path = os.path.join(self.raw_dir, "departamentos_coordenadas.csv")
        if os.path.exists(coords_path):
            df_coords = pd.read_csv(coords_path)
            df_coords['iso_a0'] = df_coords['iso_a0'].astype(str).str.strip().str.upper()
            df_coords['adm_1_name'] = df_coords['adm_1_name'].astype(str).str.strip().str.upper()

        # Agente 3: LightGBM
        agente_ml = AgentePrediccionML.cargar_modelo(self.model_dir, self.base_dir)

        # Agente 4: LSTM PyTorch
        agente_dl = AgentePrediccionDL.cargar_modelo(self.model_dir, self.base_dir)

        # Agente 5: Orquestador de Consenso (Ensemble + Alertas)
        self.orquestador = AgenteOrquestador(agente_ml, agente_dl, df_master, df_coords)

        print("SUCCESS: [SMA-ML/DL] Sistema Multi-Agente listo — Agentes 3, 4 y 5 activos.")

    # ─── Delegación al Orquestador (Agente 5) ───

    def simular_prediccion_departamento(self, iso_a0, adm_1_name, ano=None, mes=None, clima_overrides=None):
        return self.orquestador.predecir_departamento(iso_a0, adm_1_name, ano, mes, clima_overrides)

    def realizar_prediccion_vector(self, vector, iso_a0=None, adm_1_name=None, compute_shap=False):
        res = self.orquestador.agente_ml.predecir(vector, compute_shap)
        pred_ml = res["prediccion_ml"]
        result = {
            "prediccion_ml":       pred_ml,
            "riesgo_ml":           self.orquestador.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
            "prediccion_ensemble": pred_ml,
            "riesgo_ensemble":     self.orquestador.calcular_nivel_riesgo(pred_ml, iso_a0, adm_1_name),
        }
        if "shap_local" in res:
            result["shap_local"] = res["shap_local"]
        return result

    def obtener_metadatos_paises(self):
        return self.orquestador.obtener_metadatos_paises()

    def obtener_historico_departamento(self, iso_a0, adm_1_name):
        return self.orquestador.obtener_historico_departamento(iso_a0, adm_1_name)

    def obtener_top_departamentos(self, n=5):
        return self.orquestador.obtener_top_departamentos(n)
