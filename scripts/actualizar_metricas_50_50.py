"""
Actualiza metrics.json y thresholds_clasificacion.json en S3 con
ensemble 50/50 y metricas de clasificacion 3 etiquetas.
"""
import os, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             classification_report, accuracy_score, cohen_kappa_score)
import boto3

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "Base de Datos", "datos_procesados", "dataset_features_latam.csv")
TMP    = os.path.join(BASE, "Base de Datos", "modelos")

AWS_KEY    = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = "us-east-2"
BUCKET     = "epipredict-dengue"

SEQ_LEN    = 12
LSTM_FEATS = ["tmax_promedio","tmin_promedio","precipitacion",
              "humedad_promedio","agua_basica","incidencia_dengue"]
COLS_EXCLUIR = ["iso_a0","pais","adm_1_name","ano","mes",
                "casos_dengue","poblacion","incidencia_dengue"]
W_XGB, W_LSTM = 0.5, 0.5

class DengueLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def build_sequences(df, features, seq_len):
    X_list, y_list, ids = [], [], []
    for (iso, adm), grp in df.groupby(["iso_a0", "adm_1_name"]):
        grp = grp.sort_values(["ano","mes"]).reset_index(drop=True)
        vals = grp[features].values
        inc  = grp["incidencia_dengue"].values
        anos = grp["ano"].values
        meses= grp["mes"].values
        for i in range(seq_len, len(grp)):
            X_list.append(vals[i-seq_len:i])
            y_list.append(inc[i])
            ids.append((str(iso).upper(), str(adm).upper(), int(anos[i]), int(meses[i])))
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32), ids)

print("Cargando datos y modelos...")
s3 = boto3.client("s3", aws_access_key_id=AWS_KEY,
                  aws_secret_access_key=AWS_SECRET, region_name=AWS_REGION)

df = pd.read_csv(DATA)
COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
df_test = df[df["ano"] >= 2021].copy()

with open(os.path.join(TMP, "pipeline_ml.pkl"), "rb") as f:
    pipe_xgb = pickle.load(f)
with open(os.path.join(TMP, "lstm_config.json")) as f:
    cfg = json.load(f)
with open(os.path.join(TMP, "escalador_lstm.pkl"), "rb") as f:
    escalador = pickle.load(f)

model = DengueLSTMModel(6, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
model.load_state_dict(torch.load(os.path.join(TMP, "lstm_model.pth"), map_location="cpu"))
model.eval()

# Predicciones
xgb_pred_test = pipe_xgb.predict(df_test[COLS_FEAT])
xgb_lkp = {
    (str(r.iso_a0).upper(), str(r.adm_1_name).upper(), int(r.ano), int(r.mes)): float(p)
    for r, p in zip(df_test.itertuples(), xgb_pred_test)
}

X_seq, y_seq, ids_all = build_sequences(df, LSTM_FEATS, SEQ_LEN)
anos_all = np.array([i[2] for i in ids_all])
X_sc = escalador.transform(X_seq.reshape(-1,6)).reshape(X_seq.shape)

mask_test = anos_all >= 2021
with torch.no_grad():
    lstm_pred_test = model(torch.tensor(X_sc[mask_test], dtype=torch.float32)).numpy()

ids_test = [i for i in ids_all if i[2] >= 2021]

# Alinear
y_te, xgb_te, lstm_te, ids_alin = [], [], [], []
for sid, lp in zip(ids_test, lstm_pred_test):
    xp = xgb_lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[
        (df_test["iso_a0"].str.upper()==iso) &
        (df_test["adm_1_name"].str.upper()==adm) &
        (df_test["ano"]==ano) & (df_test["mes"]==mes)]
    if len(row):
        y_te.append(float(row["incidencia_dengue"].iloc[0]))
        xgb_te.append(xp); lstm_te.append(float(lp))
        ids_alin.append((iso, adm))

y_te   = np.array(y_te)
ly_te  = np.log1p(y_te)
xgb_te = np.array(xgb_te)
lstm_te= np.array(lstm_te)
print(f"  Test alineado: {len(y_te)} registros")

# Metricas regresion individuales
r2_xgb   = r2_score(ly_te, xgb_te)
mae_xgb  = mean_absolute_error(y_te, np.maximum(np.expm1(xgb_te), 0))
rmse_xgb = np.sqrt(mean_squared_error(y_te, np.maximum(np.expm1(xgb_te), 0)))

r2_lstm   = r2_score(ly_te, lstm_te)
mae_lstm  = mean_absolute_error(y_te, np.maximum(np.expm1(lstm_te), 0))
rmse_lstm = np.sqrt(mean_squared_error(y_te, np.maximum(np.expm1(lstm_te), 0)))

# Ensemble 50/50
ens_log  = W_XGB*xgb_te + W_LSTM*lstm_te
ens_real = np.maximum(np.expm1(ens_log), 0)
r2_ens   = r2_score(ly_te, ens_log)
mae_ens  = mean_absolute_error(y_te, ens_real)
rmse_ens = np.sqrt(mean_squared_error(y_te, ens_real))

# Clasificacion 3 etiquetas
df_hist = df[df["ano"] <= 2020].copy()
dept_percentiles = {}
for (iso, adm), grp in df_hist.groupby(["iso_a0", "adm_1_name"]):
    inc = grp["incidencia_dengue"].values
    dept_percentiles[(str(iso).upper(), str(adm).upper())] = (
        max(float(np.percentile(inc, 50)), 0.5),
        max(float(np.percentile(inc, 90)), 5.0),
    )
p50_g = max(float(df_hist["incidencia_dengue"].quantile(0.50)), 0.5)
p90_g = max(float(df_hist["incidencia_dengue"].quantile(0.90)), 5.0)

def clasificar(val, iso, adm):
    p50, p90 = dept_percentiles.get((iso, adm), (p50_g, p90_g))
    if val <= p50:   return 0
    elif val <= p90: return 1
    else:            return 2

y_true_cls = np.array([clasificar(v, iso, adm) for v,(iso,adm) in zip(y_te, ids_alin)])
y_pred_cls = np.array([clasificar(v, iso, adm) for v,(iso,adm) in zip(ens_real, ids_alin)])

acc   = accuracy_score(y_true_cls, y_pred_cls)
kappa = cohen_kappa_score(y_true_cls, y_pred_cls)
report = classification_report(y_true_cls, y_pred_cls,
                               target_names=["Endemico","Alerta","Epidemia"],
                               output_dict=True, zero_division=0)

dist = np.bincount(y_true_cls, minlength=3)

print(f"\n  XGBoost  R2={r2_xgb*100:.2f}%  MAE={mae_xgb:.4f}  RMSE={rmse_xgb:.4f}")
print(f"  LSTM     R2={r2_lstm*100:.2f}%  MAE={mae_lstm:.4f}  RMSE={rmse_lstm:.4f}")
print(f"  Ensemble R2={r2_ens*100:.2f}%  MAE={mae_ens:.4f}  RMSE={rmse_ens:.4f}")
print(f"  Accuracy={acc*100:.2f}%  Kappa={kappa:.4f}")
print(f"  Endemico F1={report['Endemico']['f1-score']:.4f}")
print(f"  Alerta   F1={report['Alerta']['f1-score']:.4f}")
print(f"  Epidemia F1={report['Epidemia']['f1-score']:.4f}")
print(f"  Distribucion real: Endemico={dist[0]} Alerta={dist[1]} Epidemia={dist[2]}")

# Construir metrics.json
metrics = {
    "records_procesados": int(len(df)),
    "r2_xgb":   round(float(r2_xgb),  4),
    "mae_xgb":  round(float(mae_xgb), 4),
    "rmse_xgb": round(float(rmse_xgb),4),
    "r2_lstm":   round(float(r2_lstm),  4),
    "mae_lstm":  round(float(mae_lstm), 4),
    "rmse_lstm": round(float(rmse_lstm),4),
    "r2_ensemble":   round(float(r2_ens),  4),
    "mae_ensemble":  round(float(mae_ens), 4),
    "rmse_ensemble": round(float(rmse_ens),4),
    "ensemble_w_xgb": W_XGB,
    "ensemble_w_lstm": W_LSTM,
    "acc_clasificacion":   round(float(acc),   4),
    "kappa_clasificacion": round(float(kappa), 4),
    "f1_endemico":  round(float(report["Endemico"]["f1-score"]), 4),
    "f1_alerta":    round(float(report["Alerta"]["f1-score"]),   4),
    "f1_epidemia":  round(float(report["Epidemia"]["f1-score"]), 4),
    "precision_endemico": round(float(report["Endemico"]["precision"]), 4),
    "precision_alerta":   round(float(report["Alerta"]["precision"]),   4),
    "precision_epidemia": round(float(report["Epidemia"]["precision"]), 4),
    "recall_endemico": round(float(report["Endemico"]["recall"]), 4),
    "recall_alerta":   round(float(report["Alerta"]["recall"]),   4),
    "recall_epidemia": round(float(report["Epidemia"]["recall"]), 4),
    "soporte_endemico": int(dist[0]),
    "soporte_alerta":   int(dist[1]),
    "soporte_epidemia": int(dist[2]),
    "n_train": int((df["ano"] <= 2020).sum()),
    "n_test":  int((df["ano"] >= 2021).sum()),
    "n_paises": int(df["pais"].nunique()),
    "n_departamentos": int(df["adm_1_name"].nunique()),
}

thresholds = {
    "p50_global": round(p50_g, 4),
    "p90_global": round(p90_g, 4),
    "acc_test":   round(float(acc),   4),
    "kappa_test": round(float(kappa), 4),
    "endemico_precision": round(float(report["Endemico"]["precision"]), 4),
    "endemico_recall":    round(float(report["Endemico"]["recall"]),    4),
    "endemico_f1":        round(float(report["Endemico"]["f1-score"]),  4),
    "alerta_precision":   round(float(report["Alerta"]["precision"]),   4),
    "alerta_recall":      round(float(report["Alerta"]["recall"]),      4),
    "alerta_f1":          round(float(report["Alerta"]["f1-score"]),    4),
    "epidemia_precision": round(float(report["Epidemia"]["precision"]), 4),
    "epidemia_recall":    round(float(report["Epidemia"]["recall"]),    4),
    "epidemia_f1":        round(float(report["Epidemia"]["f1-score"]),  4),
}

# Guardar local
metrics_path = os.path.join(TMP, "metrics.json")
thresh_path  = os.path.join(TMP, "thresholds_clasificacion.json")
with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
with open(thresh_path,  "w") as f: json.dump(thresholds, f, indent=2)

# Subir a S3
print("\nSubiendo a S3...")
for fname, local in [("metrics.json", metrics_path),
                     ("thresholds_clasificacion.json", thresh_path)]:
    s3.upload_file(local, BUCKET, f"modelos/{fname}")
    print(f"  OK: {fname}")

print("\nDONE")
