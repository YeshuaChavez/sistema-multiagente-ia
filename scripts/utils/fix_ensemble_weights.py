"""
Recalcula los pesos del ensemble con la nueva estrategia (proporcional al R²)
y actualiza metrics.json en disco y S3. No requiere reentrenar.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score, mean_absolute_error
sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3
from agente_4_prediccion_dl import DengueLSTMModel, _build_sequences, LSTM_SEQ_LEN

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
MODEL_DIR = os.path.join(BASE, 'Base de Datos', 'modelos')
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df_test = df[df['ano'] >= 2021].copy()
y_test  = df_test['incidencia_dengue'].values

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT    = [c for c in df.columns if c not in COLS_EXCLUIR]

# Cargar XGBoost
with open(os.path.join(MODEL_DIR, 'pipeline_ml.pkl'), 'rb') as f:
    pipe_xgb = pickle.load(f)
pred_xgb_log = pipe_xgb.predict(df_test[COLS_FEAT])
pred_xgb     = np.expm1(pred_xgb_log)
r2_xgb_log   = r2_score(np.log1p(y_test), pred_xgb_log)
mae_xgb      = mean_absolute_error(y_test, pred_xgb)
print(f"XGBoost  R²={r2_xgb_log*100:.2f}%  MAE={mae_xgb:.4f}")

# Cargar LSTM
with open(os.path.join(MODEL_DIR, 'lstm_config.json')) as f:
    cfg = json.load(f)
with open(os.path.join(MODEL_DIR, 'escalador_lstm.pkl'), 'rb') as f:
    sc_lstm = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'lstm_features.pkl'), 'rb') as f:
    lstm_feats = pickle.load(f)

lstm_model = DengueLSTMModel(cfg['input_dim'], cfg['hidden_dim'], dropout=cfg['dropout'])
lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'lstm_model.pth'), map_location='cpu'))
lstm_model.eval()

X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, lstm_feats, LSTM_SEQ_LEN)
te_mask  = anos_seq >= 2021
X_te     = X_seq[te_mask]; y_te = y_seq[te_mask]
ids_test = [seq_ids[i] for i, m in enumerate(te_mask) if m]
X_te_sc  = sc_lstm.transform(X_te.reshape(-1, len(lstm_feats))).reshape(X_te.shape)
with torch.no_grad():
    pred_lstm_log = lstm_model(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
pred_lstm    = np.expm1(pred_lstm_log)
r2_lstm_log  = r2_score(np.log1p(y_te), pred_lstm_log)
mae_lstm     = mean_absolute_error(y_te, pred_lstm)
print(f"LSTM     R²={r2_lstm_log*100:.2f}%  MAE={mae_lstm:.4f}")

# Ensemble con pesos proporcionales al R²
xgb_lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
            for r, p in zip(df_test.itertuples(), pred_xgb)}
y_l, xgb_l, lstm_l = [], [], []
for sid, lp in zip(ids_test, pred_lstm):
    xp = xgb_lkp.get(sid)
    if xp is None: continue
    iso,adm,ano,mes = sid
    row = df_test[(df_test['iso_a0'].str.upper()==iso)&(df_test['adm_1_name'].str.upper()==adm)&(df_test['ano']==ano)&(df_test['mes']==mes)]
    if len(row):
        y_l.append(float(row['incidencia_dengue'].iloc[0]))
        xgb_l.append(xp); lstm_l.append(lp)

y, xgb_a, lstm_a = np.array(y_l), np.array(xgb_l), np.array(lstm_l)
ly, lxgb, llst   = np.log1p(y), np.log1p(xgb_a), np.log1p(lstm_a)

total_r2 = r2_xgb_log + r2_lstm_log
w_xgb    = r2_xgb_log / total_r2
w_lstm   = r2_lstm_log / total_r2
ens_log  = w_xgb * lxgb + w_lstm * llst
ens_raw  = np.expm1(ens_log)
r2_ens   = r2_score(ly, ens_log)
mae_ens  = mean_absolute_error(y, ens_raw)

print(f"\n{'='*55}")
print(f"  ENSEMBLE PROPORCIONAL")
print(f"  w_xgb={w_xgb:.3f}  w_lstm={w_lstm:.3f}")
print(f"  XGBoost  : R²={r2_xgb_log*100:.2f}%")
print(f"  LSTM     : R²={r2_lstm_log*100:.2f}%")
print(f"  Ensemble : R²={r2_ens*100:.2f}%  MAE={mae_ens:.4f}")
print(f"{'='*55}")

# Actualizar metrics.json
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
with open(metrics_path) as f:
    metrics = json.load(f)

metrics['r2_xgb']          = round(r2_xgb_log, 4)
metrics['mae_xgb']         = round(mae_xgb, 4)
metrics['r2_lstm']         = round(r2_lstm_log, 4)
metrics['mae_lstm']        = round(mae_lstm, 4)
metrics['r2_ensemble']     = round(r2_ens, 4)
metrics['mae_ensemble']    = round(mae_ens, 4)
metrics['ensemble_w_xgb']  = round(w_xgb, 4)
metrics['ensemble_w_lstm'] = round(w_lstm, 4)

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
s3.upload(metrics_path, s3.PREFIX_MODELOS + 'metrics.json')
print(f"\nmetrics.json actualizado y subido a S3.")
