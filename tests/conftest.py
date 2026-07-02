# -*- coding: utf-8 -*-
"""
Configuracion compartida de pytest.

Agrega agents/ al sys.path para que los tests puedan importar los modulos
de los agentes (agente_5_alertas, agente_6_regimen, etc.) de la misma forma
en que los propios agentes se importan entre si -- mismo patron
sys.path.insert(0, ...) que usa el resto del proyecto, sin necesidad de
convertir agents/ en un paquete instalable.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS_DIR = os.path.join(ROOT, "agents")
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)
