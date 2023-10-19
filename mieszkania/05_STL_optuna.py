import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import optuna
import numpy as np
from sklearn.linear_model import LinearRegression

# Automatyczne pobieranie danych
url = "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv"
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')

# Utworzenie kolumny 'data' jako połączenie roku i kwartału
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal']*3-2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])

df = df.sort_values(by='data')
df.set_index('data', inplace=True)

data_series = df['Wartosc']

def mape(y_true, y_pred):
    return 100.0 * sum(abs((y_true - y_pred) / y_true)) / len(y_true)

def objective(trial):
    seasonal = trial.suggest_int("seasonal", 7, 25, step=2)  # tylko nieparzyste
    trend = trial.suggest_int("trend", 7, 25, step=2)  # tylko nieparzyste
    low_pass = trial.suggest_int("low_pass", 3, 25, step=2)  # tylko nieparzyste

    if low_pass <= 4:
        raise optuna.TrialPruned()  # Przycinanie prób z niską wartością low_pass

    seasonal_deg = trial.suggest_int("seasonal_deg", 0, 1)
    trend_deg = trial.suggest_int("trend_deg", 0, 1)
    low_pass_deg = trial.suggest_int("low_pass_deg", 0, 1)
    robust = trial.suggest_categorical("robust", [True, False])
    seasonal_jump = trial.suggest_int("seasonal_jump", 1, 7)
    trend_jump = trial.suggest_int("trend_jump", 1, 7)
    low_pass_jump = trial.suggest_int("low_pass_jump", 1, 7)

    res = STL(data_series, period=4, seasonal=seasonal, trend=trend, low_pass=low_pass,
              seasonal_deg=seasonal_deg, trend_deg=trend_deg, low_pass_deg=low_pass_deg,
              robust=robust, seasonal_jump=seasonal_jump, trend_jump=trend_jump,
              low_pass_jump=low_pass_jump).fit()

    y_true = data_series
    y_pred = res.trend + res.seasonal
    return mape(y_true, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Wizualizacja najlepszego wyniku
best_params = study.best_params
res_best = STL(data_series, period=4, **best_params).fit()
res_best.plot()
plt.show()
print("Najlepsze parametry:", best_params)

# Prosta ekstrapolacja składnika trendu
dates = pd.date_range(data_series.index[-1] + pd.Timedelta(days=1), end="2025-12-31", freq="Q")
X = np.array(range(len(data_series))).reshape(-1, 1)
y = res_best.trend.dropna().values
reg = LinearRegression().fit(X, y)
X_forecast = np.array(range(len(data_series), len(data_series) + len(dates))).reshape(-1, 1)
trend_forecast = reg.predict(X_forecast)

# Powtarzanie składnika sezonowości
seasonal_forecast = np.tile(res_best.seasonal[-4:], int(np.ceil(len(dates) / 4)))[:len(dates)]

# Składnik resztowy zakładamy, że będzie równy 0
resid_forecast = np.zeros(len(dates))

forecast = trend_forecast + seasonal_forecast + resid_forecast

# Rysowanie wyników
plt.figure(figsize=(10, 6))
plt.plot(data_series.index, data_series.values, label='Obserwacje')
plt.plot(dates, forecast, color='red', linestyle='--', label='Prognoza')
plt.title('Prognoza ceny za metr kwadratowy')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.show()
