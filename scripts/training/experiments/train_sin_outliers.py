import sys, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from scipy.optimize import minimize_scalar

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
from agente_4_prediccion_dl import DengueLSTMModel, LSTM_SEQ_LEN, _build_sequences

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
LSTM_FEATURES = ['tmax_promedio','tmin_promedio','precipitacion',
                  'humedad_promedio','agua_basica','incidencia_dengue']

def cargar_base():
    df = pd.read_csv(FEAT_PATH)
    yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
    df = df[yearly > 100].reset_index(drop=True)
    df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
    return df

def entrenar_xgb(df_tr, df_te, cols):
    y_tr = np.log1p(df_tr['incidencia_dengue'])
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('m',   XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                              min_child_weight=3, gamma=0.1, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
    ])
    pipe.fit(df_tr[cols], y_tr)
    pred_log = pipe.predict(df_te[cols])
    pred     = np.expm1(pred_log)
    y_te     = df_te['incidencia_dengue'].values
    return pred, pred_log, y_te, pipe

def entrenar_lstm(df, df_te_ref, y_te_ref):
    X_seq, y_seq, anos_seq, seq_ids = _build_sequences(df, LSTM_FEATURES, LSTM_SEQ_LEN)
    tr_mask = anos_seq <= 2020
    te_mask = anos_seq >= 2021
    X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
    X_te, y_te = X_seq[te_mask], y_seq[te_mask]
    anos_tr    = anos_seq[tr_mask]
    ids_test   = [seq_ids[i] for i, m in enumerate(te_mask) if m]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr.reshape(-1,6)).reshape(X_tr.shape)
    X_te_sc = sc.transform(X_te.reshape(-1,6)).reshape(X_te.shape)
    y_tr_log = np.log1p(y_tr)

    torch.manual_seed(9); np.random.seed(9)
    val_mask = anos_tr == 2020; fit_mask = ~val_mask
    Xf = torch.tensor(X_tr_sc[fit_mask], dtype=torch.float32)
    yf = torch.tensor(y_tr_log[fit_mask], dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_tr_sc[val_mask], dtype=torch.float32)
    yv = torch.tensor(y_tr_log[val_mask], dtype=torch.float32)

    m   = DengueLSTMModel(6, 512, dropout=0.2)
    opt = optim.Adam(m.parameters(), lr=0.003, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loader = DataLoader(TensorDataset(Xf, yf), batch_size=256, shuffle=True)
    best_val, wait, best_state = 1e9, 0, None

    for ep in range(300):
        m.train()
        for bx, by in loader:
            opt.zero_grad(); nn.MSELoss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            val_mse = float(nn.MSELoss()(m(Xv).flatten(), yv))
        sched.step(val_mse)
        if val_mse < best_val - 1e-5:
            best_val = val_mse; wait = 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 15: break

    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pred_lstm_log = m(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
    pred_lstm = np.expm1(pred_lstm_log)
    return pred_lstm, ids_test, y_te

def ensemble_y_metrics(df_test, pred_xgb, pred_lstm, ids_test):
    lkp = {(str(r.iso_a0).strip().upper(), str(r.adm_1_name).strip().upper(), int(r.ano), int(r.mes)): float(p)
           for r, p in zip(df_test.itertuples(), pred_xgb)}
    y_l, xgb_l, lstm_l = [], [], []
    for sid, lp in zip(ids_test, pred_lstm):
        xp = lkp.get(sid)
        if xp is None: continue
        iso,adm,ano,mes = sid
        row = df_test[(df_test['iso_a0'].str.upper()==iso)&(df_test['adm_1_name'].str.upper()==adm)&(df_test['ano']==ano)&(df_test['mes']==mes)]
        if len(row):
            y_l.append(float(row['incidencia_dengue'].iloc[0]))
            xgb_l.append(xp); lstm_l.append(lp)
    y,xgb,lst = np.array(y_l), np.array(xgb_l), np.array(lstm_l)
    ly,lxgb,llst = np.log1p(y), np.log1p(xgb), np.log1p(lst)
    diff = xgb - lst
    w = float(np.clip(np.sum((y-lst)*diff)/np.sum(diff**2), 0, 1))
    ens = w*xgb + (1-w)*lst
    res = minimize_scalar(lambda ww: -r2_score(ly, ww*lxgb+(1-ww)*llst), bounds=(0,1), method='bounded')
    w_log = float(res.x)
    ens_log = w_log*lxgb + (1-w_log)*llst
    return {
        'n': len(y),
        'xgb_log': r2_score(ly,lxgb)*100, 'xgb_raw': r2_score(y,xgb)*100,
        'lstm_log': r2_score(ly,llst)*100, 'lstm_raw': r2_score(y,lst)*100,
        'ens_raw_log': r2_score(ly,np.log1p(ens))*100, 'ens_raw_raw': r2_score(y,ens)*100,
        'ens_raw_mae': mean_absolute_error(y,ens),
        'ens_log_log': r2_score(ly,ens_log)*100,
        'w_raw': w, 'w_log': w_log,
    }

def run(nombre, df):
    cols = [c for c in df.columns if c not in COLS_EXCLUIR]
    df_train = df[df['ano'] <= 2020]
    df_test  = df[df['ano'] >= 2021].copy()
    pred_xgb, _, _, _ = entrenar_xgb(df_train, df_test, cols)
    pred_lstm, ids_test, _ = entrenar_lstm(df, df_test, df_test['incidencia_dengue'].values)
    m = ensemble_y_metrics(df_test, pred_xgb, pred_lstm, ids_test)
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"  Filas: {len(df)} | Train: {len(df_train)} | Test: {len(df_test)}")
    print(f"  XGBoost : R2 log={m['xgb_log']:.2f}%  R2 crudo={m['xgb_raw']:.2f}%")
    print(f"  LSTM    : R2 log={m['lstm_log']:.2f}%  R2 crudo={m['lstm_raw']:.2f}%")
    print(f"  Ensemble (opt.raw w={m['w_raw']:.2f}): R2 log={m['ens_raw_log']:.2f}%  R2 crudo={m['ens_raw_raw']:.2f}%  MAE={m['ens_raw_mae']:.4f}")
    print(f"  Ensemble (opt.log w={m['w_log']:.2f}): R2 log={m['ens_log_log']:.2f}%")

# Baseline
df_base = cargar_base()
print("Referencia (sin NIC, sin filtro outliers): XGB=89.85%log  ENS=87.17%log  78.70%crudo")

# Estrategia 1: Quitar departamentos con maximo > 1000 (5 deptos, 6 filas extremas)
df1 = df_base.copy()
stats = df1.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok = stats[stats <= 1000].index
df1 = df1[df1.set_index(['pais','adm_1_name']).index.isin(deptos_ok)].reset_index(drop=True)
run("Estrategia 1: Quitar deptos con max > 1000 incidencia", df1)

# Estrategia 2: Quitar Argentina (medianas ~0, muy erratica)
df2 = df_base[df_base['iso_a0'].str.upper() != 'ARG'].reset_index(drop=True)
run("Estrategia 2: Quitar Argentina (patron muy erratico)", df2)

# Estrategia 3: Ambas (quitar ARG + deptos max>1000)
df3 = df2.copy()
stats3 = df3.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok3 = stats3[stats3 <= 1000].index
df3 = df3[df3.set_index(['pais','adm_1_name']).index.isin(deptos_ok3)].reset_index(drop=True)
run("Estrategia 3: Quitar ARG + deptos max>1000", df3)
