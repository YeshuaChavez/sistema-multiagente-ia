# -*- coding: utf-8 -*-
"""
SMA-ML/DL - Sistema Multi-Agente de Predicción de Dengue
Agente 5: Alertas (Interfaz de Síntesis)
--------------------------------------------------
Responsabilidad: Unifica las predicciones del Agente 3 (XGBoost) y el Agente 4 (LSTM)
mediante un modelo de Ensemble, clasifica el escenario territorial en niveles de riesgo
y aloja la interfaz gráfica (GUI Tkinter) interactiva y de reporte técnico.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importar agentes del paquete
from agentes.agente_1_recoleccion import AgenteRecoleccion
from agentes.agente_2_preprocesamiento import AgentePreprocesamiento
from agentes.agente_3_prediccion_ml import AgentePrediccionML
from agentes.agente_4_prediccion_dl import AgentePrediccionDL

# Configuración de estética de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

def entrenar_sistema_completo():
    """
    Orquesta la ejecución secuencial de los Agentes 1, 2, 3 y 4.
    """
    # 1. Agente 1: Recolección
    recolector = AgenteRecoleccion()
    datos_crudos = recolector.ejecutar_ingesta()
    
    # 2. Agente 2: Preprocesamiento (Crea dataset_maestro_mensual_latam.csv)
    preprocesador = AgentePreprocesamiento()
    df_maestro = preprocesador.ejecutar_preprocesamiento(datos_crudos)
    
    # 3. Agente 3: Predicción ML (XGBoost + SHAP)
    agente_ml = AgentePrediccionML()
    res_ml = agente_ml.entrenar_modelo()
    
    # 4. Agente 4: Predicción DL (LSTM)
    agente_dl = AgentePrediccionDL()
    res_dl = agente_dl.entrenar_modelo()
    
    return res_ml, res_dl, df_maestro

class AgenteAlertasGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Multi-Agente SMA-ML/DL (Yeshua Chavez)")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 780)
        
        # Iniciar entrenamiento orquestado por los agentes
        self.root.update()
        try:
            print("Iniciando orquestación de Agentes Predictivos...")
            self.res_ml, self.res_dl, self.df = entrenar_sistema_completo()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Error del Sistema Multi-Agente", 
                f"Ocurrió un error al ejecutar el flujo de los agentes:\n{str(e)}"
            )
            sys.exit(1)

        # Configuración de variables de Ensemble
        self.y_test = self.res_ml['y_test']
        self.y_pred_ml = self.res_ml['y_pred']
        self.y_pred_dl = self.res_dl['y_pred']
        
        # Ensemble: Promedio de XGBoost y LSTM
        self.y_pred_ens = (self.y_pred_ml + self.y_pred_dl) / 2.0
        
        # Calcular métricas del Ensemble
        self.ens_mae = mean_absolute_error(self.y_test, self.y_pred_ens)
        self.ens_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_ens))
        self.ens_r2 = r2_score(self.y_test, self.y_pred_ens)

        # Percentiles para riesgo (calculados sobre la incidencia real del dataset mensual)
        self.p25 = self.df["incidencia_dengue"].quantile(0.25)
        self.p50 = self.df["incidencia_dengue"].quantile(0.50)
        self.p90 = self.df["incidencia_dengue"].quantile(0.90)

        # Datos para interactores
        self.COLS_FEAT = self.res_ml['cols_feat']
        self.def_val = self.df[self.COLS_FEAT].median().to_dict()
        self.sliders = {}
        self.slider_labels = {}
        self.slider_ranges = {}
        self.colores_mpl = {
            "XGBoost": "#ea580c", 
            "LSTM": "#8b5cf6", 
            "Ensemble": "#10b981",
            "Ridge": "#2563eb"  # Usado para visualizaciones adicionales
        }

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Colores
        self.style.configure("TFrame", background="#ffffff")
        self.style.configure("TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), foreground="#1e3a5f")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#64748b")
        
        # Estilo para Notebook
        self.style.configure("TNotebook", background="#f1f5f9", borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 5])
        self.style.map("TNotebook.Tab",
            background=[("selected", "#1e3a5f"), ("active", "#eff6ff"), ("!selected", "#e2e8f0")],
            foreground=[("selected", "#ffffff"), ("!selected", "#475569")]
        )

        # Treeview Styles
        self.style.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#e2e8f0", foreground="#1e293b")

    def create_widgets(self):
        # 1. Banner Superior Académico
        banner = tk.Frame(self.root, bg="#1e3a5f", height=85)
        banner.pack(fill="x", side="top")
        banner.pack_propagate(False)
        
        title_lbl = tk.Label(banner, text="🦟 Sistema Multi-Agente (SMA-ML/DL) — Predicción de Dengue en Latinoamérica",
                             font=("Segoe UI", 15, "bold"), fg="white", bg="#1e3a5f")
        title_lbl.pack(anchor="w", padx=20, pady=(12, 1))
        
        sub_lbl = tk.Label(banner, text="Sistema de Inferencia de Incidencia de Dengue en Latinoamérica  ·  Yeshua Chavez",
                            font=("Segoe UI", 9), fg="#93c5fd", bg="#1e3a5f")
        sub_lbl.pack(anchor="w", padx=20)

        # 2. Contenedor de Pestañas (Notebook Principal)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)

        # Crear pestañas principales
        self.tab_metrics = ttk.Frame(self.notebook)
        self.tab_predictor = ttk.Frame(self.notebook)
        self.tab_charts = ttk.Frame(self.notebook)
        self.tab_importance = ttk.Frame(self.notebook)
        self.tab_info = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_metrics, text="📊 Dashboard de Métricas")
        self.notebook.add(self.tab_predictor, text="🔮 Predictor en Vivo (Ensemble)")
        self.notebook.add(self.tab_charts, text="📈 Gráficos de Análisis")
        self.notebook.add(self.tab_importance, text="🔍 Explicabilidad XAI (SHAP)")
        self.notebook.add(self.tab_info, text="ℹ️ Información de Agentes")

        # Rellenar cada pestaña
        self.build_metrics_tab()
        self.build_predictor_tab()
        self.build_charts_tab()
        self.build_importance_tab()
        self.build_info_tab()

    # =============================================================================
    # PESTAÑA 1 — DASHBOARD DE MÉTRICAS DE REGRESIÓN
    # =============================================================================
    def build_metrics_tab(self):
        main_frame = ttk.Frame(self.tab_metrics, padding=15)
        main_frame.pack(fill="both", expand=True)

        # Fila superior de KPIs
        kpis_frame = ttk.Frame(main_frame)
        kpis_frame.pack(fill="x", pady=(0, 15))

        mejor_nombre = "LSTM" if self.res_dl['test_r2'] > self.res_ml['test_r2'] else "XGBoost"
        mejor_r2 = max(self.res_dl['test_r2'], self.res_ml['test_r2'])

        kpis = [
            ("TOTAL REGISTROS", f"{self.df.shape[0]:,}", "observaciones mensuales"),
            ("PARTICIÓN CRONOLÓGICA", f"{self.res_ml['n_train']:,} / {self.res_ml['n_test']:,}", "Train (14-20) / Test (21-22)"),
            ("MEJOR MODELO (TEST)", f"{mejor_nombre}", f"R² de {mejor_r2 * 100:.2f}%"),
            ("MÉTRICAS EVALUADAS", "R², MAE, RMSE", "regresión supervisada")
        ]

        for i, (label, val, unit) in enumerate(kpis):
            card = tk.Frame(kpis_frame, bg="#f8fafc", highlightbackground="#cbd5e1", highlightthickness=1, bd=0)
            card.grid(row=0, column=i, padx=8, sticky="nsew")
            kpis_frame.columnconfigure(i, weight=1)
            
            lbl_title = tk.Label(card, text=label, font=("Segoe UI", 8, "bold"), fg="#64748b", bg="#f8fafc")
            lbl_title.pack(pady=(10, 1))
            lbl_val = tk.Label(card, text=val, font=("Segoe UI", 14, "bold"), fg="#1e293b", bg="#f8fafc")
            lbl_val.pack(pady=1)
            lbl_unit = tk.Label(card, text=unit, font=("Segoe UI", 8), fg="#94a3b8", bg="#f8fafc")
            lbl_unit.pack(pady=(0, 10))

        # Sección Regresión
        reg_lbl = ttk.Label(main_frame, text="📋 Métricas de Regresión de los Agentes Predictivos", font=("Segoe UI", 11, "bold"))
        reg_lbl.pack(anchor="w", pady=(8, 3))

        cols_reg = ("Modelo/Agente", "Tipo de Modelo", "CV MAE (14-20) ↓", "CV RMSE (14-20) ↓", "CV R² (14-20) ↑", "Test MAE (21-22) ↓", "Test RMSE (21-22) ↓", "Test R² (21-22) ↑")
        tree_reg = ttk.Treeview(main_frame, columns=cols_reg, show="headings", height=3)
        tree_reg.pack(fill="x", pady=(0, 15))

        for col in cols_reg:
            tree_reg.heading(col, text=col)
            tree_reg.column(col, anchor="center", width=125)

        # Fila 1: XGBoost (Agente 3)
        tree_reg.insert("", "end", values=(
            "Agente 3: XGBoost", "Machine Learning (Ensamble)",
            f"{self.res_ml['cv_mae']:.4f}", f"{self.res_ml['cv_rmse']:.4f}", f"{self.res_ml['cv_r2']*100:.2f}%",
            f"{self.res_ml['test_mae']:.4f}", f"{self.res_ml['test_rmse']:.4f}", f"{self.res_ml['test_r2']*100:.2f}%"
        ))
        
        # Fila 2: LSTM (Agente 4)
        tree_reg.insert("", "end", values=(
            "Agente 4: LSTM", "Deep Learning (Recurrente)",
            f"{self.res_dl['cv_mae']:.4f}", f"{self.res_dl['cv_rmse']:.4f}", f"{self.res_dl['cv_r2']*100:.2f}%",
            f"{self.res_dl['test_mae']:.4f}", f"{self.res_dl['test_rmse']:.4f}", f"{self.res_dl['test_r2']*100:.2f}%"
        ))

        # Fila 3: Ensemble Promedio (Consolidación Agente 5)
        tree_reg.insert("", "end", values=(
            "Agente 5: Ensemble", "Consolidación (Promedio)",
            "N/A", "N/A", "N/A",
            f"{self.ens_mae:.4f}", f"{self.ens_rmse:.4f}", f"{self.ens_r2*100:.2f}%"
        ))

        # Cuadro informativo aclaratorio
        info_frame = tk.Frame(main_frame, bg="#eff6ff", highlightbackground="#bfdbfe", highlightthickness=1)
        info_frame.pack(fill="x", pady=5)
        
        info_txt = (
            "ℹ️ NOTA METODOLÓGICA SÓLIDA:\n"
            "Este sistema utiliza una partición cronológica (Entrenamiento: 2014-2020, Prueba: 2021-2022) para evaluar el desempeño "
            "del modelo en el tiempo sin riesgo de fuga de información temporal.\n\n"
            "Observación Clave: Los modelos predicen directamente la tasa de incidencia de dengue (casos por 100k hab.). Durante El Niño (2021-2022), "
            "la incidencia se disparó de forma anómala. Los modelos de caja negra basados en árboles (Random Forest y XGBoost) tienen un "
            "'límite de extrapolación' intrínseco, mientras que la LSTM captura las dependencias secuenciales no lineales a largo plazo, logrando "
            "asilar la inercia temporal para una predicción robusta."
        )
        tk.Label(info_frame, text=info_txt, font=("Segoe UI", 9), fg="#1e40af", bg="#eff6ff", justify="left", wraplength=1200).pack(padx=12, pady=8)

    # =============================================================================
    # PESTAÑA 2 — PREDICTOR EN TIEMPO REAL
    # =============================================================================
    def build_predictor_tab(self):
        main_frame = ttk.Frame(self.tab_predictor, padding=10)
        main_frame.pack(fill="both", expand=True)

        left_inputs = ttk.Frame(main_frame, padding=5)
        left_inputs.grid(row=0, column=0, sticky="nsew")
        
        right_outputs = ttk.Frame(main_frame, padding=10)
        right_outputs.grid(row=0, column=1, sticky="ns")
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, minsize=380)
        main_frame.rowconfigure(0, weight=1)

        # Canvas con scroll vertical
        canvas = tk.Canvas(left_inputs, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_inputs, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, padding=5)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        #Selector por País
        ref_frame = tk.Frame(scroll_frame, bg="#f8fafc", highlightbackground="#e2e8f0", highlightthickness=1)
        ref_frame.pack(fill="x", pady=(5, 12), padx=5, ipady=8)
        
        ref_lbl = tk.Label(ref_frame, text="📍 CARGAR VALORES TÍPICOS DE UN PAÍS (MEDIANA HISTÓRICA)", 
                           font=("Segoe UI", 9, "bold"), fg="#1e3a5f", bg="#f8fafc")
        ref_lbl.pack(anchor="w", padx=15, pady=(8, 2))
        
        self.paises = ["— Ingresar manualmente —"] + sorted(self.df["pais"].unique().tolist())
        self.country_combo = ttk.Combobox(ref_frame, values=self.paises, state="readonly", font=("Segoe UI", 10))
        self.country_combo.set("— Ingresar manualmente —")
        self.country_combo.pack(anchor="w", padx=15, fill="x", expand=True)
        self.country_combo.bind("<<ComboboxSelected>>", self.on_country_select)

        # Columnas de sliders
        cols_frame = ttk.Frame(scroll_frame)
        cols_frame.pack(fill="both", expand=True, padx=5)

        col1_clima = ttk.Frame(cols_frame, padding=5)
        col1_clima.grid(row=0, column=0, sticky="n")
        
        col2_agua = ttk.Frame(cols_frame, padding=5)
        col2_agua.grid(row=0, column=1, sticky="n")
        
        col3_lags = ttk.Frame(cols_frame, padding=5)
        col3_lags.grid(row=0, column=2, sticky="n")

        cols_frame.columnconfigure(0, weight=1)
        cols_frame.columnconfigure(1, weight=1)
        cols_frame.columnconfigure(2, weight=1)

        # Columna 1: Clima e Incidencia Lags
        tk.Label(col1_clima, text="🌡️ Clima Actual", font=("Segoe UI", 10, "bold"), fg="#1e293b").pack(anchor="w", pady=(5, 8))
        clima_params = [
            ("tmax_promedio", "Temperatura Máxima (°C)", 11.0, 40.0),
            ("tmin_promedio", "Temperatura Mínima (°C)", -3.0, 28.0),
            ("precipitacion", "Precipitación (mm)", 0.0, 600.0),
            ("humedad_promedio", "Humedad Relativa (%)", 20.0, 95.0),
        ]
        for key, desc, lo, hi in clima_params:
            self.crear_slider(col1_clima, key, desc, lo, hi)

        tk.Label(col1_clima, text="⏱️ Rezagos de Incidencia", font=("Segoe UI", 10, "bold"), fg="#1e293b").pack(anchor="w", pady=(15, 8))
        incidencia_params = [
            ("incidencia_lag1", "Incidencia lag 1 mes (casos/100k)", 0.0, 75.0),
            ("incidencia_lag2", "Incidencia lag 2 mes (casos/100k)", 0.0, 75.0),
            ("incidencia_lag3", "Incidencia lag 3 mes (casos/100k)", 0.0, 75.0),
        ]
        for key, desc, lo, hi in incidencia_params:
            self.crear_slider(col1_clima, key, desc, lo, hi)

        # Columna 2: Agua JMP
        tk.Label(col2_agua, text="💧 Acceso al Agua (Censos/JMP)", font=("Segoe UI", 10, "bold"), fg="#1e293b").pack(anchor="w", pady=(5, 8))
        self.crear_slider(col2_agua, "agua_basica", "Agua Básica (%)", 68.0, 100.0)

        # Columna 3: Rezagos Climáticos
        tk.Label(col3_lags, text="⏱️ Rezagos Climáticos (Lags 1-3)", font=("Segoe UI", 10, "bold"), fg="#1e293b").pack(anchor="w", pady=(5, 8))
        lag_params = [
            ("tmax_lag1", "Tmax lag 1 mes (°C)", 11.0, 40.0),
            ("tmax_lag2", "Tmax lag 2 mes (°C)", 11.0, 40.0),
            ("tmax_lag3", "Tmax lag 3 mes (°C)", 11.0, 40.0),
            ("tmin_lag1", "Tmin lag 1 mes (°C)", -3.0, 28.0),
            ("tmin_lag2", "Tmin lag 2 mes (°C)", -3.0, 28.0),
            ("tmin_lag3", "Tmin lag 3 mes (°C)", -3.0, 28.0),
            ("precipitacion_lag1", "Prec lag 1 mes (mm)", 0.0, 600.0),
            ("precipitacion_lag2", "Prec lag 2 mes (mm)", 0.0, 600.0),
            ("precipitacion_lag3", "Prec lag 3 mes (mm)", 0.0, 600.0),
            ("humedad_lag1", "Humedad lag 1 mes (%)", 20.0, 95.0),
            ("humedad_lag2", "Humedad lag 2 mes (%)", 20.0, 95.0),
            ("humedad_lag3", "Humedad lag 3 mes (%)", 20.0, 95.0),
        ]
        for key, desc, lo, hi in lag_params:
            self.crear_slider(col3_lags, key, desc, lo, hi)

        # Panel de Outputs Derecho
        title_out = tk.Label(right_outputs, text="📡 PREDICCIONES DEL MODELO", font=("Segoe UI", 11, "bold"), fg="#1e293b")
        title_out.pack(pady=(5, 10))

        self.cards = {}
        model_info = [
            ("XGBoost", "🟠 XGBoost Regressor (Agente 3)", "#fff7ed", "#ea580c"),
            ("LSTM", "🟣 LSTM PyTorch (Agente 4)", "#f5f3ff", "#8b5cf6"),
        ]

        for key, title, bg_col, border_col in model_info:
            frame_c = tk.Frame(right_outputs, bg=bg_col, bd=1, relief="solid", highlightbackground=border_col)
            frame_c.pack(fill="x", pady=6, ipady=4)
            
            lbl_m_title = tk.Label(frame_c, text=title, font=("Segoe UI", 9, "bold"), fg="#374151", bg=bg_col)
            lbl_m_title.pack(anchor="w", padx=12, pady=(6, 1))
            
            val_lbl = tk.Label(frame_c, text="0.00", font=("Segoe UI", 18, "bold"), fg=border_col, bg=bg_col)
            val_lbl.pack(anchor="w", padx=12)
            
            unit_lbl = tk.Label(frame_c, text="casos por 100,000 hab.", font=("Segoe UI", 8), fg="#6b7280", bg=bg_col)
            unit_lbl.pack(anchor="w", padx=12)
            
            risk_lbl = tk.Label(frame_c, text="Normal", font=("Segoe UI", 9, "bold"), bg="#e2e8f0", fg="#334155", width=26)
            risk_lbl.pack(pady=(4, 6))

            self.cards[key] = (val_lbl, risk_lbl)

        # Card de Ensemble Promedio (Agente 5)
        frame_e = tk.Frame(right_outputs, bg="#f1f5f9", bd=2, relief="groove")
        frame_e.pack(fill="x", pady=10, ipady=5)
        
        lbl_e_title = tk.Label(frame_e, text="🤖 PREDICCIÓN PROMEDIO ENSEMBLE (Agente 5)", font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#f1f5f9")
        lbl_e_title.pack(anchor="w", padx=12, pady=(6, 1))
        
        self.ens_val_lbl = tk.Label(frame_e, text="0.00", font=("Segoe UI", 20, "bold"), fg="#10b981", bg="#f1f5f9")
        self.ens_val_lbl.pack(anchor="w", padx=12)
        
        unit_lbl_e = tk.Label(frame_e, text="casos por 100,000 hab.", font=("Segoe UI", 8), fg="#64748b", bg="#f1f5f9")
        unit_lbl_e.pack(anchor="w", padx=12)
        
        self.ens_risk_lbl = tk.Label(frame_e, text="Normal", font=("Segoe UI", 9, "bold"), bg="#cbd5e1", fg="#1e293b", width=25)
        self.ens_risk_lbl.pack(pady=(4, 6))

        # Iniciar predicciones
        self.actualizar_predicciones()

    def crear_slider(self, parent, key, label_text, lo, hi):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3, padx=2)

        lbl = ttk.Label(frame, text=label_text, width=32, anchor="w", font=("Segoe UI", 8))
        lbl.pack(side="top", anchor="w")

        val_var = tk.DoubleVar()
        init_val = float(round(self.def_val.get(key, (lo + hi) / 2), 4))
        val_var.set(init_val)

        sub_frame = ttk.Frame(frame)
        sub_frame.pack(fill="x")

        val_lbl = ttk.Label(sub_frame, text=f"{init_val:.2f}", width=7, anchor="e", font=("Segoe UI", 8))
        
        slider = ttk.Scale(
            sub_frame, from_=lo, to=hi, variable=val_var, orient="horizontal",
            command=lambda e, kv=key, vv=val_var, vl=val_lbl: self.on_slider_move(kv, vv, vl)
        )
        slider.pack(side="left", fill="x", expand=True, padx=(0, 5))
        val_lbl.pack(side="right")

        self.sliders[key] = val_var
        self.slider_labels[key] = val_lbl
        self.slider_ranges[key] = (lo, hi)

    def on_slider_move(self, key, var, label):
        val = var.get()
        label.config(text=f"{val:.2f}")
        self.actualizar_predicciones()

    def on_country_select(self, event):
        pais_sel = self.country_combo.get()
        if pais_sel == "— Ingresar manualmente —":
            val_ref = self.df[self.COLS_FEAT].median().to_dict()
        else:
            val_ref = self.df[self.df["pais"] == pais_sel][self.COLS_FEAT].median().to_dict()
        
        for key, var in self.sliders.items():
            if key in val_ref:
                lo, hi = self.slider_ranges[key]
                val = float(val_ref[key])
                val_clamped = max(lo, min(hi, val))
                var.set(val_clamped)
                self.slider_labels[key].config(text=f"{val_clamped:.2f}")
                
        self.actualizar_predicciones()

    def actualizar_predicciones(self):
        # 1. Crear vector de entrada
        vector = []
        for col in self.COLS_FEAT:
            if col in self.sliders:
                vector.append(self.sliders[col].get())
            else:
                vector.append(self.def_val[col])

        entrada = np.array([vector])
        
        # 2. Preprocesamiento (Imputación y escalado de XGBoost)
        entrada_imp_ml = self.res_ml['imputador'].transform(entrada)
        entrada_esc_ml = self.res_ml['escalador'].transform(entrada_imp_ml)
        
        # 3. Predicción XGBoost
        pred_ml = float(self.res_ml['modelo'].predict(entrada_esc_ml)[0])
        pred_ml = max(0.0, pred_ml)
        
        # 4. Predicción LSTM
        # Preprocesar datos usando imputador/escalador de la LSTM
        entrada_imp_dl = self.res_dl['imputador'].transform(entrada)
        entrada_esc_dl = self.res_dl['escalador'].transform(entrada_imp_dl)
        df_row = pd.DataFrame(entrada_esc_dl, columns=self.COLS_FEAT)
        
        # Convertir a formato temporal (3, 5) y static (5)
        seq_t, static_t = self.res_dl['preparar_secuencias_fn'](df_row)
        
        # Evaluar LSTM
        self.res_dl['modelo'].eval()
        with torch.no_grad():
            pred_dl = float(self.res_dl['modelo'](seq_t, static_t).numpy()[0][0])
        pred_dl = max(0.0, pred_dl)

        # 5. Ensemble (Agente 5)
        pred_ens = (pred_ml + pred_dl) / 2.0

        # Actualizar Tarjetas
        preds = {"XGBoost": pred_ml, "LSTM": pred_dl}
        for name, val in preds.items():
            val_lbl, risk_lbl = self.cards[name]
            val_lbl.config(text=f"{val:.4f}")
            self.aplicar_badge_riesgo(val, risk_lbl)
            
        self.ens_val_lbl.config(text=f"{pred_ens:.4f}")
        self.aplicar_badge_riesgo(pred_ens, self.ens_risk_lbl)

    def aplicar_badge_riesgo(self, pred, widget):
        if pred <= self.p25:
            widget.config(text="🟢 Riesgo: Bajo / Normal", bg="#dcfce7", fg="#166534")
        elif pred <= self.p50:
            widget.config(text="🟡 Riesgo: Vigilancia", bg="#fefce8", fg="#854d0e")
        elif pred <= self.p90:
            widget.config(text="🟠 Riesgo: Alerta", bg="#fff7ed", fg="#9a3412")
        else:
            widget.config(text="🔴 Riesgo: Epidemia", bg="#fef2f2", fg="#991b1b")

    # =============================================================================
    # PESTAÑA 3 — GRÁFICOS DE ANÁLISIS
    # =============================================================================
    def build_charts_tab(self):
        main_frame = ttk.Frame(self.tab_charts, padding=5)
        main_frame.pack(fill="both", expand=True)

        self.charts_notebook = ttk.Notebook(main_frame)
        self.charts_notebook.pack(fill="both", expand=True)

        self.subtab_real_pred = ttk.Frame(self.charts_notebook)
        self.subtab_compare = ttk.Frame(self.charts_notebook)
        self.subtab_residuals = ttk.Frame(self.charts_notebook)

        self.charts_notebook.add(self.subtab_real_pred, text="📍 Reales vs. Predichos")
        self.charts_notebook.add(self.subtab_compare, text="📊 Comparativa de Métricas")
        self.charts_notebook.add(self.subtab_residuals, text="📉 Distribución de Residuos")

        self.build_real_pred_chart(self.subtab_real_pred)
        self.build_compare_chart(self.subtab_compare)
        self.build_residuals_chart(self.subtab_residuals)

    def build_real_pred_chart(self, parent):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), facecolor="#ffffff")
        y_test_arr = np.array(self.y_test)
        lim = y_test_arr.max() * 1.05
        
        model_preds = {
            "XGBoost": self.y_pred_ml,
            "LSTM": self.y_pred_dl,
            "Ensemble": self.y_pred_ens
        }
        
        for ax, (nombre, pred) in zip(axes, model_preds.items()):
            color = self.colores_mpl[nombre]
            mae = mean_absolute_error(self.y_test, pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred))
            r2 = r2_score(self.y_test, pred)
            
            ax.plot([0, lim], [0, lim], color="#94a3b8", lw=1.5, ls="--")
            ax.scatter(y_test_arr, pred, color=color, alpha=0.45, s=15, linewidths=0)
            ax.text(0.05, 0.95,
                    f"MAE  = {mae:.3f}\nRMSE = {rmse:.3f}\nR²   = {r2*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1, alpha=0.9))
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Incidencia real (casos/100k)", fontsize=7.5)
            ax.set_ylabel("Incidencia predicha (casos/100k)", fontsize=7.5)
            ax.set_title(f"{nombre}", fontsize=8.5, fontweight="bold", color="#1e293b")
            ax.tick_params(labelsize=7)
            ax.grid(True, ls=":", alpha=0.4, color="#cbd5e1")
            
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def build_compare_chart(self, parent):
        nombres = ["XGBoost", "LSTM", "Ensemble"]
        colores = [self.colores_mpl[n] for n in nombres]
        x = np.arange(len(nombres))
        w = 0.4
        
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), facecolor="#ffffff")
        
        # MAE
        vals_mae = [self.res_ml['test_mae'], self.res_dl['test_mae'], self.ens_mae]
        axes[0].bar(x, vals_mae, w, color=colores, edgecolor="white")
        axes[0].set_title("Test MAE ↓ (Menor es mejor)", fontsize=8.5, fontweight="bold")
        
        # RMSE
        vals_rmse = [self.res_ml['test_rmse'], self.res_dl['test_rmse'], self.ens_rmse]
        axes[1].bar(x, vals_rmse, w, color=colores, edgecolor="white")
        axes[1].set_title("Test RMSE ↓ (Menor es mejor)", fontsize=8.5, fontweight="bold")
        
        # R2
        vals_r2 = [self.res_ml['test_r2']*100, self.res_dl['test_r2']*100, self.ens_r2*100]
        axes[2].bar(x, vals_r2, w, color=colores, edgecolor="white")
        axes[2].set_title("Test R² % ↑ (Mayor es mejor)", fontsize=8.5, fontweight="bold")
        
        for idx, ax in enumerate(axes):
            ax.set_xticks(x)
            ax.set_xticklabels(nombres, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", ls=":", alpha=0.4, color="#cbd5e1")
            
            # Poner etiquetas de valor
            for i, val in enumerate([vals_mae, vals_rmse, vals_r2][idx]):
                ax.text(i, val + (val*0.015), f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def build_residuals_chart(self, parent):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), facecolor="#ffffff")
        y_test_arr = np.array(self.y_test)
        
        model_preds = {
            "XGBoost": self.y_pred_ml,
            "LSTM": self.y_pred_dl,
            "Ensemble": self.y_pred_ens
        }
        
        for ax, (nombre, pred) in zip(axes, model_preds.items()):
            color = self.colores_mpl[nombre]
            residuos = pred - y_test_arr
            
            ax.hist(residuos, bins=25, color=color, alpha=0.70, edgecolor="white", lw=0.5)
            ax.axvline(0, color="#1e293b", lw=1.2, ls="--", label="Residuo = 0")
            ax.axvline(residuos.mean(), color="#f59e0b", lw=1.2, label=f"Media = {residuos.mean():.2f}")
            
            ax.set_title(f"{nombre}", fontsize=8.5, fontweight="bold", color="#1e293b")
            ax.set_xlabel("Residuo (predicho − real)", fontsize=7.5)
            ax.set_ylabel("Frecuencia", fontsize=7.5)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, ls=":", alpha=0.4, color="#cbd5e1")
            
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # =============================================================================
    # PESTAÑA 4 — IMPORTANCIA & EXPLICABILIDAD (SHAP VALUES)
    # =============================================================================
    def build_importance_tab(self):
        main_frame = ttk.Frame(self.tab_importance, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="#ffffff")
        
        # Graficar importancias globales SHAP
        shap_imp = self.res_ml['shap_importance']
        ax.barh(shap_imp.index[::-1], shap_imp.values[::-1], color="#ea580c", alpha=0.8)
        ax.set_title("Explicabilidad Global de Variables del Agente Predictivo (TreeSHAP Values)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importancia Promedio Absoluta |SHAP Value|", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", ls=":", alpha=0.4, color="#cbd5e1")
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # =============================================================================
    # PESTAÑA 5 — INFORMACIÓN DE LOS AGENTES
    # =============================================================================
    def build_info_tab(self):
        main_frame = ttk.Frame(self.tab_info, padding=15)
        main_frame.pack(fill="both", expand=True)

        info_box = ttk.Frame(main_frame, padding=10)
        info_box.pack(fill="both", expand=True)

        detalles = (
            "👤 AUTOR:\n"
            "Yeshua Chavez\n\n"
            "🤖 ARQUITECTURA CONCEPTUAL COOPERATIVA SMA-ML/DL:\n"
            "El sistema se compone de 5 agentes informáticos especializados que cooperan de manera secuencial y asíncrona:\n\n"
            "1. AGENTE DE RECOLECCIÓN (Agente 1):\n"
            "   * Ingesta de forma dirigida casos de dengue (OpenDengue), clima satelital (NASA POWER), agua básica (JMP) y "
            "población a partir de Censos Gubernamentales locales.\n\n"
            "2. AGENTE DE PREPROCESAMIENTO (Agente 2):\n"
            "   * Encargado del cruzamiento y normalización de la tasa de incidencia (casos por 100k hab.).\n"
            "   * Aplica rezagos temporales (lags 1, 2 y 3 meses) simétricos para clima e incidencia, y genera el dataset maestro "
            "'dataset_maestro_mensual_latam.csv'.\n\n"
            "3. AGENTE DE PREDICCIÓN MACHINE LEARNING (Agente 3):\n"
            "   * Entrena el algoritmo de ensamble XGBoost Regressor.\n"
            "   * Integra una capa nativa de explicabilidad algorítmica (XAI) mediante el cálculo de valores SHAP (Shapley Additive exPlanations).\n\n"
            "4. AGENTE DE PREDICCIÓN DEEP LEARNING (Agente 4):\n"
            "   * Implementa una red neuronal recurrente LSTM (Long Short-Term Memory) en PyTorch para asimilar dinámicas secuenciales a escala subnacional.\n\n"
            "5. AGENTE DE ALERTAS Y SÍNTESIS (Agente 5):\n"
            "   * Unifica los pronósticos de XGBoost y LSTM en una salida robusta promedio (Ensemble).\n"
            "   * Clasifica los escenarios epidemiológicos territoriales en 4 niveles de riesgo (Normal, Vigilancia, Alerta, Epidemia) y provee "
            "la consola interactiva."
        )

        txt_widget = tk.Text(info_box, font=("Segoe UI", 10), wrap="word", bg="#f8fafc", relief="flat")
        txt_widget.insert("1.0", detalles)
        txt_widget.config(state="disabled")
        
        scroll_txt = ttk.Scrollbar(info_box, orient="vertical", command=txt_widget.yview)
        txt_widget.configure(yscrollcommand=scroll_txt.set)
        
        txt_widget.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scroll_txt.pack(side="right", fill="y")


if __name__ == "__main__":
    root = tk.Tk()
    app = AgenteAlertasGUI(root)
    root.mainloop()
