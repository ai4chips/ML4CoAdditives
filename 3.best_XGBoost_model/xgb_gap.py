import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import os
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import multiprocessing
import joblib
import pickle

path = os.path.dirname(os.path.abspath(__file__))
with open(path+'/Input/gap_29.pkl', 'rb') as file:
    x = pickle.load(file)

with open(path + '/Input/EnergyGap.pkl', 'rb') as file:
    y = pickle.load(file)
y = np.array(y).reshape(-1, 1).ravel()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.00012, 0.3),

    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    r2 = r2_score(y_val, y_val_pred)
    return r2

n_processes = multiprocessing.cpu_count()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, n_jobs=-1)


best_params = study.best_trial.params
best_r2 = study.best_value
print('Best params:', best_params)
print('Best val_R2:', best_r2)


path1 = '/Output/xgb_gap_optimization_history.csv'
model_file = path + path1
history_data = study.trials_dataframe()
history_data.to_csv(model_file, index=False)

model = XGBRegressor(**best_params, random_state=42)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

y_test_pred = model.predict(x_test)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

y_val_pred = model.predict(x_val)
r2_val = r2_score(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)


print("train R2:", r2_train)
print("test R2:", r2_test)
print("validation R2:", r2_val)
print("train MAE", mae_train)
print("test MAE", mae_test)
print("validation MAE", mae_val)

model_file = path + '/Output/gap_xgb_bayes.m'
joblib.dump(model, model_file)