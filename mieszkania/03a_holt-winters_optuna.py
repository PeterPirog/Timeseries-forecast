import pandas as pd
import matplotlib.pyplot as plt
import optuna
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Wczytanie danych
url = 'https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv'
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal'] * 3 - 2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values(by='data')
data_series = df.set_index('data')['Wartosc']
data_series.index.freq = "QS-OCT"


def mape(y_true, y_pred):
    return 100.0 * sum(abs((y_true - y_pred) / y_true)) / len(y_true)


def objective(trial):
    trend = trial.suggest_categorical("trend", [None, "add", "mul"])
    if trend is None:
        damped_trend = False
    else:
        damped_trend = trial.suggest_categorical("damped_trend", [True, False])

    #damped_trend = trial.suggest_categorical("damped_trend", [True, False])
    seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
    seasonal_periods = 4
    initialization_method = trial.suggest_categorical("initialization_method",
                                                      ["estimated", "heuristic", "legacy-heuristic"])
    use_boxcox = trial.suggest_categorical("use_boxcox", [True, False, "lambda"])
    if use_boxcox == "lambda":
        lambda_val = trial.suggest_float("lambda_val", 0.01, 2)
    else:
        lambda_val = use_boxcox

    model = ExponentialSmoothing(data_series,
                                 trend=trend,
                                 damped_trend=damped_trend,
                                 seasonal=seasonal,
                                 seasonal_periods=seasonal_periods,
                                 initialization_method=initialization_method,
                                 use_boxcox=lambda_val)
    model_fit = model.fit(optimized=True, use_brute=True)

    y_pred = model_fit.fittedvalues
    return mape(data_series, y_pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

# Utworzenie i wytrenowanie modelu Holta-Wintersa z najlepszymi hiperparametrami
best_params = study.best_params
lambda_val = best_params.pop("lambda_val", None)

if best_params["use_boxcox"] == "lambda":
    best_params["use_boxcox"] = lambda_val

model = ExponentialSmoothing(data_series, **best_params, seasonal_periods=4).fit()

forecast = model.forecast(10)

# Wizualizacja prognozy
plt.figure(figsize=(12, 6))
plt.plot(data_series.index, data_series.values, label='Historical')
plt.plot(forecast.index, forecast.values, color='red', label='Forecast')
plt.title('Prognoza ceny za metr kwadratowy przy użyciu modelu Holta-Wintersa')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.show()

print("Najlepsze parametry:", best_params)
