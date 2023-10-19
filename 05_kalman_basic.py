import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np

# Wczytanie i przetworzenie danych
def resample_data(filename, frequency='H'):
    data = pd.read_csv(filename, delimiter=",")
    data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
    data.set_index('Date Time', inplace=True)
    data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
    data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)
    resampled_data = data.resample(frequency).mean()
    return resampled_data

# Kalman Filter do prognozowania wartości "T (degC)"
# Kalman Filter do prognozowania wartości "T (degC)"
def kalman_filter_temperature(data):
    # Dodanie niewielkiego szumu
    noisy_data = data['T (degC)'].values + 1e-5 * np.random.randn(data['T (degC)'].shape[0])

    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(noisy_data, n_iter=10)
    (filtered_state_means, _) = kf.filter(noisy_data)
    return filtered_state_means


filename = 'tools/jena_climate_2009_2016.csv'
hourly_data = resample_data(filename, frequency='1H')

print(hourly_data['T (degC)'].isnull().sum())
hourly_data['T (degC)'] = hourly_data['T (degC)'].interpolate()


filtered_temperatures = kalman_filter_temperature(hourly_data)

# Wyświetlanie wyników
plt.figure(figsize=(12, 6))
plt.plot(hourly_data.index, hourly_data['T (degC)'], label="Original Data")
plt.plot(hourly_data.index, filtered_temperatures, label="Kalman Filter Predictions", linestyle="--")
plt.title("Temperature Prediction using Kalman Filter")
plt.legend()
plt.show()
