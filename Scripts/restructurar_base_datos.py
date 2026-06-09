# -*- coding: utf-8 -*-
"""
Script de Restructuración de Base de Datos
-----------------------------------------
Organiza los archivos CSV en subcarpetas 'datos_crudos' y 'datos_procesados'.
"""

import os
import shutil

base_dir = r"c:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final"
db_dir = os.path.join(base_dir, "Base de Datos")

# Nuevas carpetas
raw_dir = os.path.join(db_dir, "datos_crudos")
raw_pob = os.path.join(raw_dir, "poblacion")
proc_dir = os.path.join(db_dir, "datos_procesados")
proc_pob = os.path.join(proc_dir, "poblacion")

# Crear directorios si no existen
os.makedirs(raw_pob, exist_ok=True)
os.makedirs(proc_pob, exist_ok=True)

# 1. Archivos en la raíz de Base de Datos a mover a raw/
raw_files = [
    "Temporal_extract_V1_3.csv",
    "agua_jmp_crudo.csv",
    "clima_nasa_crudo.csv",
    "departamentos_areas.csv",
    "departamentos_coordenadas.csv"
]

for filename in raw_files:
    src = os.path.join(db_dir, filename)
    dst = os.path.join(raw_dir, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Movido a crudos: {filename}")

# 2. Archivo maestro procesado a mover a datos_procesados/
proc_files = [
    "dataset_maestro_mensual_latam.csv"
]

for filename in proc_files:
    src = os.path.join(db_dir, filename)
    dst = os.path.join(proc_dir, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Movido a procesados: {filename}")

# 3. Mover censos crudos e interpolados
pob_original = os.path.join(db_dir, "poblacion")
if os.path.exists(pob_original):
    for filename in os.listdir(pob_original):
        src = os.path.join(pob_original, filename)
        if filename.startswith("censos_crudos_"):
            dst = os.path.join(raw_pob, filename)
            shutil.move(src, dst)
            print(f"Censo crudo movido: {filename}")
        elif filename.startswith("poblacion_"):
            dst = os.path.join(proc_pob, filename)
            shutil.move(src, dst)
            print(f"Población procesada movida: {filename}")
            
    # Eliminar directorio original vacío
    try:
        os.rmdir(pob_original)
        print("Eliminada carpeta de población original vacía.")
    except Exception as e:
        print(f"No se pudo eliminar carpeta original: {e}")

print("¡Estructura de Base de Datos reorganizada con éxito!")
