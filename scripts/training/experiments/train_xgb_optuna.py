import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')

BASE      = r'C:\Users\yeshu\Documents\Inteligencia Artificial\Proyecto Final'
FEAT_PATH = os.path.join(BASE, 'Base de Datos', 'datos_procesados', 'dataset_features_latam.csv')

df = pd.read_csv(FEAT_PATH)
yearly = df.groupby(['pais','ano'])['casos_dengue'].transform('sum')
df = df[yearly > 100].reset_index(drop=True)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)

# Quitar 5 deptos con pico > 1000
stats = df.groupby(['pais','adm_1_name'])['incidencia_dengue'].max()
deptos_ok = stats[stats <= 1000].index
df = df[df.set_index(['pais','adm_1_name']).index.isin(deptos_ok)].reset_index(drop=True)

# Opcion 1: Añadir lags climaticos 7-12
print("Calculando lags climaticos 7-12...")
for lag in range(7, 13):
    for col in ['tmax_promedio','tmin_promedio','precipitacion','humedad_promedio']:
        nombre = col.replace('_promedio','') if '_promedio' in col else col
        df[f'{nombre}_lag{lag}'] = df.groupby(['pais','adm_1_name'])[col].shift(lag)

COLS_EXCLUIR = ['iso_a0','pais','adm_1_name','ano','mes','casos_dengue','poblacion','incidencia_dengue']
COLS_FEAT    = [c for c in df.columns if c not in COLS_EXCLUIR]

df_train = df[df['ano'] <= 2020]
df_test  = df[df['ano'] >= 2021].copy()
X_train  = df_train[COLS_FEAT]
y_train  = np.log1p(df_train['incidencia_dengue'])
X_test   = df_test[COLS_FEAT]
y_test   = df_test['incidencia_dengue'].values

print(f"Features: {len(COLS_FEAT)} | Train: {len(df_train)} | Test: {len(df_test)}")

imp = SimpleImputer(strategy='median')
sc  = StandardScaler()
X_tr_imp = sc.fit_transform(imp.fit_transform(X_train))
X_te_imp = sc.transform(imp.transform(X_test))

tscv = TimeSeriesSplit(n_splits=4)

# Opcion 2: Bayesian Optimization con Optuna
def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 600, 1500),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma':            trial.suggest_float('gamma', 0.0, 0.3),
        'subsample':        trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 2.0),
    }
    xgb = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    scores = cross_val_score(xgb, X_tr_imp, y_train, cv=tscv, scoring='r2', n_jobs=1)
    return -scores.mean()

print("Optuna: 80 trials...")
study = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=80, show_progress_bar=False)

best = study.best_params
print(f"Mejor R2 CV: {-study.best_value*100:.2f}%")
print("Params optimos:")
for k, v in sorted(best.items()):
    print(f"  {k:22s}: {v}")

# Entrenar final
xgb_final = XGBRegressor(**best, random_state=42, n_jobs=-1, verbosity=0)
xgb_final.fit(X_tr_imp, y_train)
pred_log = xgb_final.predict(X_te_imp)
pred     = np.expm1(pred_log)

r2_log = r2_score(np.log1p(y_test), pred_log)
r2_raw = r2_score(y_test, pred)
mae    = mean_absolute_error(y_test, pred)

print(f"\nXGBoost (Optuna + lags clima 12):")
print(f"  R2 log  : {r2_log*100:.2f}%")
print(f"  R2 crudo: {r2_raw*100:.2f}%")
print(f"  MAE     : {mae:.4f}")
print(f"  (antes sin outliers: 89.75% log)")
