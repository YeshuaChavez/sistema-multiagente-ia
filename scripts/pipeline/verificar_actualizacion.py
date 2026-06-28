# -*- coding: utf-8 -*-
"""
verificar_actualizacion.py — DenguePredict SMA-ML/DL
=====================================================
FASE 10 DEL CICLO DE VIDA: Monitoreo y Mantenimiento
-----------------------------------------------------
Implementa la etapa de mantenimiento continuo del SMA-ML/DL:

  10a — Deteccion de nueva version de datos:
        Consulta la API de GitHub para obtener el SHA del ultimo commit
        en la carpeta data/releases del repositorio OpenDengue.
        Si el SHA difiere del guardado localmente, hay datos nuevos disponibles.

  10b — Deteccion de drift de covariables (PSI — Population Stability Index):
        Descarga datos climaticos recientes de NASA POWER para una muestra
        representativa de departamentos (1 por pais) y calcula el PSI para
        cada feature climatica respecto a la distribucion de entrenamiento.
        PSI < 0.1  → estable    (sin accion)
        PSI 0.1-0.2 → moderado  (monitorear)
        PSI >= 0.2  → alto      (priorizar reentrenamiento)

  10c — Reentrenamiento automatico (con --retrain):
        Si hay nueva version, descarga el dataset, ejecuta el pipeline completo
        de entrenamiento (agentes 2, 3, 4) y sube los modelos a S3.

  Nota: El drift de concepto (cambio en la relacion features→target) no puede
  evaluarse automaticamente porque OpenDengue publica datos con 6-12 meses de
  latencia respecto al tiempo real. Solo el drift de covariables es detectable
  en tiempo casi-real via variables climaticas de NASA POWER.

Ejecutado automaticamente el 1ro de cada mes via GitHub Actions (.github/workflows/retrain.yml).

Uso manual:
    python scripts/pipeline/verificar_actualizacion.py
    python scripts/pipeline/verificar_actualizacion.py --retrain    # reentrenar si hay nueva version
    python scripts/pipeline/verificar_actualizacion.py --drift-only # solo calcular drift
"""

import os, sys, json, argparse, hashlib, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

GITHUB_API      = "https://api.github.com/repos/OpenDengue/master-repo/commits"
OPENDENGUE_FILE = "data/releases"   # monitorear carpeta de releases para detectar nuevas versiones
VERSION_FILE  = os.path.join(ROOT, "data", "models", "data_version.json")
METRICS_FILE  = os.path.join(ROOT, "data", "models", "metrics.json")
DRIFT_FILE    = os.path.join(ROOT, "data", "models", "drift_report.json")
DENGUE_LOCAL  = os.path.join(ROOT, "data", "raw", "Temporal_extract_V1_3.csv")
FEATURES_FILE = os.path.join(ROOT, "data", "processed", "dataset_features_latam.csv")

# Features climáticas para monitorear drift (las de mayor importancia SHAP)
CLIMATE_FEATURES = ["tmax_promedio", "tmin_promedio", "precipitacion", "humedad_promedio"]

# ────────────────────────────────────────────────
# 1. VERIFICACIÓN DE NUEVA VERSIÓN EN GITHUB
# ────────────────────────────────────────────────

def obtener_sha_github():
    """Obtiene el SHA del último commit que tocó el archivo OpenDengue en GitHub."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(
            GITHUB_API,
            params={"path": OPENDENGUE_FILE, "per_page": 1},
            headers=headers,
            timeout=15,
        )
        if r.status_code == 200:
            commits = r.json()
            if commits:
                sha = commits[0]["sha"][:12]
                fecha = commits[0]["commit"]["committer"]["date"]
                return sha, fecha
        print(f"   [GitHub API] HTTP {r.status_code} — sin SHA disponible")
    except Exception as e:
        print(f"   [GitHub API] Error de conexión: {e}")
    return None, None


def sha_local():
    """Lee el SHA guardado del último dataset descargado."""
    if not os.path.exists(VERSION_FILE):
        return None
    with open(VERSION_FILE) as f:
        return json.load(f).get("sha_github")


def guardar_version(sha, fecha_github, sha_archivo):
    os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
    data = {
        "sha_github":     sha,
        "fecha_github":   fecha_github,
        "sha_archivo_md5": sha_archivo,
        "verificado_en":  datetime.now(timezone.utc).isoformat(),
    }
    with open(VERSION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   [Versión] Guardada en {VERSION_FILE}")


def md5_archivo(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 512), b""):
            h.update(chunk)
    return h.hexdigest()


def hay_nueva_version():
    sha_remoto, fecha = obtener_sha_github()
    if sha_remoto is None:
        print("   [Versión] No se pudo consultar GitHub — asumiendo sin cambios.")
        return False, None, None
    sha_guardado = sha_local()
    print(f"   [Versión] SHA remoto : {sha_remoto}  ({fecha})")
    print(f"   [Versión] SHA guardado: {sha_guardado or 'ninguno'}")
    nueva = sha_remoto != sha_guardado
    return nueva, sha_remoto, fecha


# ────────────────────────────────────────────────
# 2. DRIFT DE COVARIABLES (PSI sobre clima NASA)
# ────────────────────────────────────────────────

def calcular_psi(ref: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre distribución de referencia y actual."""
    eps = 1e-6
    breakpoints = np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts,    _ = np.histogram(ref,    bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    ref_pct    = ref_counts    / (ref_counts.sum()    + eps)
    actual_pct = actual_counts / (actual_counts.sum() + eps)

    psi = np.sum((actual_pct - ref_pct) * np.log((actual_pct + eps) / (ref_pct + eps)))
    return float(round(psi, 4))


def nivel_drift(psi: float) -> str:
    if psi < 0.1:   return "estable"
    if psi < 0.2:   return "moderado"
    return "alto"


def distribucion_referencia():
    """Calcula estadísticas de referencia desde el set de entrenamiento (años ≤ 2020)."""
    if not os.path.exists(FEATURES_FILE):
        print("   [Drift] No se encontró dataset de features para referencia.")
        return None
    df = pd.read_csv(FEATURES_FILE, usecols=["ano"] + CLIMATE_FEATURES)
    df_train = df[df["ano"] <= 2020]
    ref = {}
    for feat in CLIMATE_FEATURES:
        vals = df_train[feat].dropna().values
        ref[feat] = vals
    return ref


def obtener_clima_reciente(n_meses: int = 12):
    """
    Descarga datos climáticos recientes de NASA POWER para una muestra de
    departamentos representativos (centros geográficos de cada país).
    Solo para detección de drift — no reemplaza el dataset completo.
    """
    coords_path = os.path.join(ROOT, "data", "raw", "departamentos_coordenadas.csv")
    if not os.path.exists(coords_path):
        print("   [Drift] No hay caché de coordenadas — omitiendo drift de clima.")
        return None

    df_coords = pd.read_csv(coords_path)
    # Tomar 1 departamento por país (el primero) para agilizar
    muestra = df_coords.groupby("iso_a0").first().reset_index()

    ano_actual = datetime.now().year
    ano_inicio = ano_actual - 1  # últimos ~12 meses

    registros = []
    for _, row in muestra.iterrows():
        url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        params = {
            "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
            "community": "RE",
            "longitude": row["lon"],
            "latitude":  row["lat"],
            "start":     str(ano_inicio),
            "end":       str(ano_actual),
            "format":    "JSON",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                props = r.json()["properties"]["parameter"]
                for clave, tmax in props["T2M_MAX"].items():
                    if clave.endswith("13"):
                        continue
                    registros.append({
                        "tmax_promedio":    tmax,
                        "tmin_promedio":    props["T2M_MIN"][clave],
                        "precipitacion":    props["PRECTOTCORR"][clave],
                        "humedad_promedio": props["RH2M"][clave],
                    })
        except Exception as e:
            print(f"   [Drift] NASA POWER falló para {row['iso_a0']}: {e}")

    return pd.DataFrame(registros).replace(-999.0, np.nan) if registros else None


def detectar_drift():
    """Calcula PSI para cada feature climática y guarda el reporte (siempre)."""
    print("\n[Drift] Calculando drift de covariables (clima NASA POWER)...")
    os.makedirs(os.path.dirname(DRIFT_FILE), exist_ok=True)

    ref = distribucion_referencia()
    if ref is None:
        reporte = {
            "verificado_en": datetime.now(timezone.utc).isoformat(),
            "estado": "sin_datos_referencia",
            "features": {},
            "psi_max": None,
            "alerta_drift": False,
            "nota": "No se encontró dataset_features_latam.csv para calcular referencia PSI.",
        }
        with open(DRIFT_FILE, "w") as f:
            json.dump(reporte, f, indent=2)
        print(f"   [Drift] Reporte guardado (sin referencia): {DRIFT_FILE}")
        return reporte

    print("   [Drift] Descargando datos climáticos recientes (muestra por país)...")
    df_actual = obtener_clima_reciente()
    if df_actual is None or df_actual.empty:
        reporte = {
            "verificado_en": datetime.now(timezone.utc).isoformat(),
            "estado": "sin_datos_recientes",
            "features": {},
            "psi_max": None,
            "alerta_drift": False,
            "nota": "NASA POWER no devolvió datos recientes. Drift no calculado; se revisará en el próximo ciclo.",
        }
        with open(DRIFT_FILE, "w") as f:
            json.dump(reporte, f, indent=2)
        print(f"   [Drift] Reporte guardado (NASA POWER sin datos): {DRIFT_FILE}")
        return reporte

    resultados = {}
    print(f"\n   {'Feature':<25} {'PSI':>8}  {'Nivel'}")
    print(f"   {'─'*25} {'─'*8}  {'─'*10}")
    for feat in CLIMATE_FEATURES:
        vals_actual = df_actual[feat].dropna().values
        if len(vals_actual) < 5:
            continue
        psi = calcular_psi(ref[feat], vals_actual)
        nivel = nivel_drift(psi)
        simbolo = "✓" if nivel == "estable" else ("⚠" if nivel == "moderado" else "✗")
        print(f"   {feat:<25} {psi:>8.4f}  {simbolo} {nivel}")
        resultados[feat] = {"psi": psi, "nivel": nivel}

    psi_max = max(v["psi"] for v in resultados.values()) if resultados else 0.0
    alerta_drift = psi_max >= 0.2

    reporte = {
        "verificado_en": datetime.now(timezone.utc).isoformat(),
        "estado": "calculado",
        "features": resultados,
        "psi_max": round(psi_max, 4),
        "alerta_drift": alerta_drift,
        "nota": (
            "Drift detectado en features de entrada (covariable shift). "
            "El drift de concepto no puede evaluarse hasta disponer de nuevos "
            "datos etiquetados de OpenDengue." if alerta_drift else
            "Distribución de features climáticas estable respecto al período de entrenamiento."
        ),
    }

    with open(DRIFT_FILE, "w") as f:
        json.dump(reporte, f, indent=2)
    print(f"\n   [Drift] Reporte guardado en {DRIFT_FILE}")
    print(f"   [Drift] PSI máximo: {psi_max:.4f} → {'⚠ ALERTA' if alerta_drift else '✓ Estable'}")
    return reporte


# ────────────────────────────────────────────────
# 3. PIPELINE COMPLETO
# ────────────────────────────────────────────────

def actualizar_trained_at():
    """Añade trained_at y data_version a metrics.json."""
    if not os.path.exists(METRICS_FILE):
        return
    with open(METRICS_FILE) as f:
        m = json.load(f)
    version_info = {}
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE) as f:
            version_info = json.load(f)
    m["trained_at"]      = datetime.now(timezone.utc).isoformat()
    m["data_sha_github"] = version_info.get("sha_github", "desconocido")
    m["data_fecha"]      = version_info.get("fecha_github", "desconocida")
    with open(METRICS_FILE, "w") as f:
        json.dump(m, f, indent=2)
    print(f"   [Métricas] trained_at y data_version actualizados en metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Verificar actualización OpenDengue + drift climático")
    parser.add_argument("--retrain",    action="store_true", help="Reentrenar si hay nueva versión")
    parser.add_argument("--drift-only", action="store_true", help="Solo calcular drift, sin chequear versión")
    args = parser.parse_args()

    print("=" * 65)
    print("  DenguePredict — Verificación de Actualización y Drift")
    print("=" * 65)

    reentrenar = False

    if not args.drift_only:
        print("\n[1/2] Verificando versión del dataset OpenDengue en GitHub...")
        nueva, sha_remoto, fecha_github = hay_nueva_version()

        if nueva:
            print(f"\n   *** Nueva versión detectada ({sha_remoto}) ***")
            print("   Descargando dataset actualizado...")
            from agente_1_recoleccion import AgenteRecoleccion
            a = AgenteRecoleccion(base_dir=ROOT)
            if os.path.exists(DENGUE_LOCAL):
                os.remove(DENGUE_LOCAL)
            a._descargar_opendengue()
            sha_md5 = md5_archivo(DENGUE_LOCAL)
            guardar_version(sha_remoto, fecha_github, sha_md5)
            reentrenar = True
        else:
            print("   Dataset OpenDengue sin cambios desde la última verificación.")
            if os.path.exists(DENGUE_LOCAL) and sha_remoto:
                guardar_version(sha_remoto, fecha_github, md5_archivo(DENGUE_LOCAL))

    print("\n[2/2] Detectando drift de covariables climáticas...")
    reporte_drift = detectar_drift()
    drift_alto = reporte_drift and reporte_drift.get("alerta_drift", False)

    if drift_alto:
        print("\n   ⚠  Drift alto en features climáticas.")
        print("   Nota: el drift de concepto no puede evaluarse sin nuevos datos de OpenDengue.")

    print("\n" + "─" * 65)
    if reentrenar and args.retrain:
        print("  → Iniciando reentrenamiento completo (nueva versión OpenDengue)...")
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "training", "entrenar_modelos.py")],
            cwd=ROOT,
        )
        if result.returncode == 0:
            actualizar_trained_at()
            import s3_client as s3
            s3.upload(METRICS_FILE,  s3.PREFIX_MODELOS + "metrics.json")
            s3.upload(DRIFT_FILE,    s3.PREFIX_MODELOS + "drift_report.json")
            s3.upload(VERSION_FILE,  s3.PREFIX_MODELOS + "data_version.json")
            print("  ✓ Reentrenamiento completado y artefactos subidos a S3.")
        else:
            print("  ✗ El reentrenamiento falló — revisar logs.")
    elif reentrenar:
        print("  → Nueva versión disponible. Ejecuta con --retrain para reentrenar.")
    else:
        print("  ✓ Sin cambios en el dataset. Sin reentrenamiento necesario.")
        if drift_alto:
            print("  ⚠  Drift climático alto registrado en drift_report.json.")

    print("=" * 65)


if __name__ == "__main__":
    main()
