import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import multiprocessing
import joblib
import pickle

path = os.path.dirname(os.path.abspath(__file__))
with open(path+'/Input/gap_29.pkl', 'rb') as file:
    x = pickle.load(file).astype(np.float32)

with open(path + '/Input/EnergyGap.pkl', 'rb') as file:
    y = pickle.load(file)
y = np.array(y, dtype=np.float32).reshape(-1, 1).ravel()
#x=x.iloc[:50, :]
#y=y[:50,]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 80, 200),
        'max_depth': trial.suggest_int('max_depth', 25, 45),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
    }
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42,
        ccp_alpha = 0.01
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    r2 = r2_score(y_val, y_val_pred)
    return r2



#n_processes = multiprocessing.cpu_count()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)


best_params = study.best_trial.params
best_r2 = study.best_value
print('Best params:', best_params)
print('Best val_r2:', best_r2)

# 保存每一次贝叶斯优化后的值

#path1 = '/Output/rdkit_optimization_history.csv'
#model_file = path + path1
#history_data = study.trials_dataframe()
#history_data.to_csv(model_file, index=False)


# 使用训练集上的模型进行预测和评估
model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
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
print("最终模型")
print("训练集 R2:", r2_train)
print("测试集 R2:", r2_test)
print("验证集 R2:", r2_val)
print("训练集MAE",mae_train)
print("测试集MAE",mae_test)
print("验证集MAE",mae_val)

#model_file = path + '/Output/gap_rf_bayes.m'
#joblib.dump(model, model_file)
import pandas as pd
df = pd.read_excel(os.path.join(path, 'Output', 'gap随机森林调参.xlsx'))

