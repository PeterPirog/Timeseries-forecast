import pandas as pd
import optuna
from statsmodels.tsa.arima.model import ARIMA

# Wczytanie danych
url = 'https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv'
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal'] * 3 - 2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values(by='data')
data_series = df.set_index('data')['Wartosc']
data_series.index.freq = "QS-OCT"


# Funkcja celu dla Optuna
def objective(trial):
    p = trial.suggest_int('p', 0, 7)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 7)

    P = trial.suggest_int('P', 0, 5)
    D = trial.suggest_int('D', 0, 2)
    Q = trial.suggest_int('Q', 0, 5)
    s = 4  # dla danych kwartalnych

    trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])

    enforce_stationarity = trial.suggest_categorical('enforce_stationarity', [True, False])
    enforce_invertibility = trial.suggest_categorical('enforce_invertibility', [True, False])
    concentrate_scale = trial.suggest_categorical('concentrate_scale', [True, False])

    try:
        model = ARIMA(data_series, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend,
                      enforce_stationarity=enforce_stationarity,
                      enforce_invertibility=enforce_invertibility,
                      concentrate_scale=concentrate_scale)
        model_fit = model.fit()
        return model_fit.aic
    except:
        return float('inf')


# Optymalizacja z użyciem Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Wyświetlenie wyników
print('\nNajlepsze parametry ARIMA:', study.best_params)
print('Najlepszy AIC:', study.best_value)
