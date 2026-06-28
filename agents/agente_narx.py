# -*- coding: utf-8 -*-
"""
Agente NARX — Nonlinear AutoRegressive with eXogenous inputs (MLP, 73 features)
=================================================================================
NARX define la estructura del input:
  y(t) = f( y(t-1),..,y(t-12), u(t), u(t-1),..,u(t-m) )
           ^-- AR (incidencia lags) --^  ^-- exogenas (clima, vecinos) --^

Las 73 features del dataset ya contienen ambos terminos:
  - incidencia_lag1..lag12            -> parte autorregresiva (AR)
  - tmax/tmin/precip lags, vecinos... -> parte exogena (X)

La funcion no-lineal f se implementa con un MLP (red feedforward multicapa).
Comparacion directa con XGBoost: mismo input (73 features), distinto modelo.

Ciclo de vida completo:
  Fase 1  — Carga de datos y separacion train/test
  Fase 2  — Normalizacion (StandardScaler en train)
  Fase 3  — Transformacion log1p del target
  Fase 4  — Baseline MLP (1 capa oculta 64, lr=0.01, 40 epocas)
  Fase 5  — Evaluacion baseline
  Fase 6  — Bayesian Optimization (Optuna/TPE) + TimeSeriesSplit K=5 folds
             30 trials x espacio continuo de hiperparametros
  Fase 7  — Reentrenamiento con mejores hiperparametros (early stopping)
  Fase 8  — Evaluacion final en test set 2021-2022 (escala log1p)
  Fase 9  — Serializacion y subida a AWS S3
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dotenv import load_dotenv

load_dotenv('.env')

SEMILLA = 42
N_TRIALS = 30
COLS_EXCLUIR = [
    'iso_a0', 'pais', 'adm_1_name', 'ano', 'mes',
    'casos_dengue', 'poblacion', 'incidencia_dengue'
]


class NARXModel(nn.Module):
    """MLP de N capas con BatchNorm y Dropout — implementa la funcion f de NARX."""
    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _entrenar_mlp(X_tr, y_tr_log, hidden_layers, lr, dropout, epochs, seed):
    _set_seed(seed)
    model = NARXModel(X_tr.shape[1], hidden_layers, dropout)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr_log, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        sched.step(loss)
    return model


class AgenteNARX:
    def __init__(self, base_dir='.'):
        self.base_dir  = base_dir
        self.model_dir = os.path.join(base_dir, 'data', 'models')
        self.proc_dir  = os.path.join(base_dir, 'data', 'processed')

    def entrenar_modelo(self):
        print('=' * 70)
        print('  ENTRENANDO — AGENTE NARX-MLP (73 features + Bayesian Optimization)')
        print('=' * 70)

        # ── Fase 1: Carga ─────────────────────────────────────────────────────
        print('\n   [Fase 1] Cargando dataset...')
        df = pd.read_csv(os.path.join(self.proc_dir, 'dataset_features_latam.csv'))

        COLS_FEAT = [c for c in df.columns if c not in COLS_EXCLUIR]
        print(f'   Features NARX: {len(COLS_FEAT)}  (identicas a XGBoost)')

        df_train = df[df['ano'] <= 2020].copy().reset_index(drop=True)
        df_test  = df[df['ano'] >= 2021].copy().reset_index(drop=True)

        X_train_raw = df_train[COLS_FEAT].values.astype(np.float32)
        y_train_raw = df_train['incidencia_dengue'].values.astype(np.float32)
        anos_train  = df_train['ano'].values

        X_test_raw  = df_test[COLS_FEAT].values.astype(np.float32)
        y_test_raw  = df_test['incidencia_dengue'].values.astype(np.float32)

        print(f'   Train: {len(X_train_raw)} registros | Test: {len(X_test_raw)} registros')

        # ── Fase 2: Normalizacion ─────────────────────────────────────────────
        print('\n   [Fase 2] Normalizando (StandardScaler sobre train)...')
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc  = scaler.transform(X_test_raw)

        # ── Fase 3: Log1p del target ──────────────────────────────────────────
        y_train_log = np.log1p(y_train_raw)
        y_test_log  = np.log1p(y_test_raw)

        # ── Fase 4-5: Baseline ────────────────────────────────────────────────
        print('\n   [Fase 4] Baseline MLP (64 neuronas, lr=0.01, 40 epocas)...')
        _set_seed(SEMILLA)
        model_bl = _entrenar_mlp(X_train_sc, y_train_log, (64,), 0.01, 0.0, 40, SEMILLA)
        model_bl.eval()
        with torch.no_grad():
            pred_bl_log = model_bl(
                torch.tensor(X_test_sc, dtype=torch.float32)).numpy()
        r2_bl  = r2_score(y_test_log, pred_bl_log)
        mae_bl = mean_absolute_error(y_test_raw, np.expm1(pred_bl_log))
        print(f'   [Fase 5] Baseline — R²={r2_bl*100:.2f}%  MAE={mae_bl:.4f}')

        # ── Fase 6: Bayesian Optimization con Optuna ──────────────────────────
        print(f'\n   [Fase 6] Bayesian Optimization (Optuna/TPE) — {N_TRIALS} trials x 5 folds...')
        folds = [
            (anos_train <= 2016, anos_train == 2016),
            (anos_train <= 2016, anos_train == 2017),
            (anos_train <= 2017, anos_train == 2018),
            (anos_train <= 2018, anos_train == 2019),
            (anos_train <= 2019, anos_train == 2020),
        ]

        def objective(trial):
            n_layers   = trial.suggest_int('n_layers', 1, 3)
            hidden_layers = tuple(
                trial.suggest_int(f'h_{i}', 32, 512, log=True)
                for i in range(n_layers)
            )
            lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.4)

            r2_folds = []
            for tr_mask, val_mask in folds:
                if tr_mask.sum() == 0 or val_mask.sum() == 0:
                    continue
                sc_f = StandardScaler()
                X_tr_f = sc_f.fit_transform(X_train_sc[tr_mask])
                X_vl_f = sc_f.transform(X_train_sc[val_mask])
                y_tr   = y_train_log[tr_mask]
                y_vl   = y_train_log[val_mask]

                m = _entrenar_mlp(X_tr_f, y_tr, hidden_layers, lr, dropout, 80, SEMILLA)
                m.eval()
                with torch.no_grad():
                    pred_v = m(torch.tensor(X_vl_f, dtype=torch.float32)).numpy()
                r2_folds.append(r2_score(y_vl, pred_v))

            return float(np.mean(r2_folds))

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEMILLA),
        )
        trial_log = []

        def callback(study, trial):
            h = tuple(
                trial.params.get(f'h_{i}', '-')
                for i in range(trial.params.get('n_layers', 1))
            )
            print(f'   Trial {trial.number+1:02d}/{N_TRIALS}  '
                  f'hidden={h}  lr={trial.params.get("lr", 0):.5f}  '
                  f'dropout={trial.params.get("dropout", 0):.2f}  '
                  f'R2_CV={trial.value*100:.2f}%'
                  f'{"  <-- mejor" if trial.value == study.best_value else ""}')
            trial_log.append({
                'trial': trial.number,
                'params': dict(trial.params),
                'r2_cv': trial.value,
            })

        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

        best = study.best_trial
        n_layers_best = best.params['n_layers']
        hidden_best   = tuple(best.params[f'h_{i}'] for i in range(n_layers_best))
        lr_best       = best.params['lr']
        dropout_best  = best.params['dropout']

        print(f'\n   Mejor trial: hidden={hidden_best}  lr={lr_best:.5f}  '
              f'dropout={dropout_best:.2f}  -> R2_CV={best.value*100:.2f}%')

        # ── Fase 7: Reentrenamiento completo (early stopping) ─────────────────
        print(f'\n   [Fase 7] Reentrenando con mejores hiperparametros (300 epocas + early stopping)...')
        _set_seed(SEMILLA)
        model_best = NARXModel(X_train_sc.shape[1], hidden_best, dropout_best)
        opt    = torch.optim.Adam(model_best.parameters(), lr=lr_best, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        loss_fn = nn.MSELoss()
        X_tr_t  = torch.tensor(X_train_sc, dtype=torch.float32)
        y_tr_t  = torch.tensor(y_train_log, dtype=torch.float32)

        best_loss, patience_cnt, best_sd = np.inf, 0, None
        for ep in range(300):
            model_best.train()
            opt.zero_grad()
            loss = loss_fn(model_best(X_tr_t), y_tr_t)
            loss.backward()
            opt.step()
            sched.step(loss)
            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                best_sd   = {k: v.clone() for k, v in model_best.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= 25:
                print(f'   Early stopping en epoca {ep+1}')
                break

        if best_sd:
            model_best.load_state_dict(best_sd)

        # ── Fase 8: Evaluacion final ──────────────────────────────────────────
        print('\n   [Fase 8] Evaluando en test set 2021-2022...')
        model_best.eval()
        with torch.no_grad():
            pred_log = model_best(
                torch.tensor(X_test_sc, dtype=torch.float32)).numpy()

        pred_real = np.maximum(np.expm1(pred_log), 0)

        r2_narx   = r2_score(y_test_log, pred_log)
        mae_narx  = mean_absolute_error(y_test_raw, pred_real)
        rmse_narx = np.sqrt(mean_squared_error(y_test_raw, pred_real))

        print(f'\n{"="*70}')
        print(f'  NARX-MLP  ({len(COLS_FEAT)} features, mismo input que XGBoost)')
        print(f'  Arquitectura : MLP {hidden_best}')
        print(f'  Dropout      : {dropout_best:.2f}')
        print(f'  LR           : {lr_best:.5f}')
        print(f'  R²  = {r2_narx*100:.2f}%  (escala log1p)')
        print(f'  MAE = {mae_narx:.2f}  casos/100k')
        print(f'  RMSE= {rmse_narx:.2f}')
        print(f'{"="*70}')

        try:
            with open(os.path.join(self.model_dir, 'metrics.json')) as f:
                m = json.load(f)
            print(f'\n  Comparacion de modelos:')
            print(f'  XGBoost  R²={m["r2_xgb"]*100:.2f}%  MAE={m["mae_xgb"]:.2f}  (73 feat, arboles)')
            print(f'  LSTM     R²={m["r2_lstm"]*100:.2f}%  MAE={m["mae_lstm"]:.2f}  (6 feat, recurrente)')
            print(f'  NARX-MLP R²={r2_narx*100:.2f}%  MAE={mae_narx:.2f}  (73 feat, red neuronal) <--')
        except Exception:
            pass

        # ── Fase 9: Guardar y subir a S3 ─────────────────────────────────────
        print('\n   [Fase 9] Guardando artefactos...')
        narx_cfg = {
            'input_dim':     len(COLS_FEAT),
            'hidden_layers': list(hidden_best),
            'dropout':       round(dropout_best, 4),
            'lr':            round(lr_best, 6),
            'r2':            round(float(r2_narx), 4),
            'mae':           round(float(mae_narx), 4),
            'rmse':          round(float(rmse_narx), 4),
            'r2_baseline':   round(float(r2_bl), 4),
            'r2_cv_best':    round(float(best.value), 4),
            'n_features':    len(COLS_FEAT),
            'n_trials':      N_TRIALS,
            'optimizer':     'Optuna/TPE',
            'features':      COLS_FEAT,
            'trials':        trial_log,
        }

        narx_model_path  = os.path.join(self.model_dir, 'narx_model.pth')
        narx_config_path = os.path.join(self.model_dir, 'narx_config.json')
        narx_scaler_path = os.path.join(self.model_dir, 'escalador_narx.pkl')

        torch.save(model_best.state_dict(), narx_model_path)
        with open(narx_config_path, 'w') as f:
            json.dump(narx_cfg, f, indent=2)
        with open(narx_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        try:
            import boto3
            from dotenv import dotenv_values
            env = dotenv_values('.env')
            s3 = boto3.client(
                's3',
                aws_access_key_id=env.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=env.get('AWS_SECRET_ACCESS_KEY'),
                region_name=env.get('AWS_DEFAULT_REGION', 'us-east-1')
            )
            bucket = env.get('S3_BUCKET_NAME', 'epipredict-dengue')
            for local, key in [
                (narx_model_path,  'modelos/narx_model.pth'),
                (narx_config_path, 'modelos/narx_config.json'),
                (narx_scaler_path, 'modelos/escalador_narx.pkl'),
            ]:
                s3.upload_file(local, bucket, key)
                print(f'   Subido: {key}')
        except Exception as e:
            print(f'   S3 upload omitido: {e}')

        print('\nSUCCESS: [Agente NARX-MLP] entrenado y guardado.')
        return narx_cfg


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agente = AgenteNARX(base_dir=base)
    agente.entrenar_modelo()
