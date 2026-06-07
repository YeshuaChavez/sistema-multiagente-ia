# -*- coding: utf-8 -*-
"""
UNMSM | Trabajo de Grado | SMA-ML/DL
Lanzador Principal (main.py)
--------------------------------------------------
Responsabilidad: Punto de entrada principal para ejecutar el Sistema Multi-Agente (SMA-ML/DL).
Inicializa la GUI del Agente 5, la cual orquesta la ejecución secuencial de la recolección,
preprocesamiento y modelamiento.
"""

import sys
import tkinter as tk
from agentes.agente_5_alertas import AgenteAlertasGUI

# Configurar encoding para consola
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    # Banner
    print("      SISTEMA MULTI-AGENTE DE REGRESIÓN DE DENGUE (SMA-ML/DL)")
    print("      Universidad Nacional Mayor de San Marcos (UNMSM)")
    print("      Tesis de Grado  ·  Yeshua Chavez")
    print("=" * 70)
    
    root = tk.Tk()
    app = AgenteAlertasGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
