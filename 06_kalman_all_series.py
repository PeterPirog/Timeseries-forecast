import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np
from sklearn.metrics import mean_absolute_error

def resample_data(filename, frequency='H'):
    data = pd.read_csv(filename, delimiter=",")
    data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
    data.set_index('Date Time', inplace=True)
    data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
    data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)
    resampled_data = data.resample(frequency).mean()
    return resampled_data

def kalman_filter_multivariate(data):
    noisy_data = data + 1e-5 * np.random.randn(*data.shape)

    n_dim_state = data.shape[1]
    kf = KalmanFilter(initial_state_mean=np.zeros(n_dim_state), n_dim_obs=n_dim_state)
    kf = kf.em(noisy_data, n_iter=10)
    (filtered_state_means, _) = kf.filter(noisy_data)

    # Zapis macierzy do plików
    np.savetxt('transition_matrices.csv', kf.transition_matrices, delimiter=',')
    np.savetxt('observation_matrices.csv', kf.observation_matrices, delimiter=',')
    np.savetxt('transition_covariance.csv', kf.transition_covariance, delimiter=',')
    np.savetxt('observation_covariance.csv', kf.observation_covariance, delimiter=',')
    np.savetxt('initial_state_mean.csv', kf.initial_state_mean, delimiter=',')
    np.savetxt('initial_state_covariance.csv', kf.initial_state_covariance, delimiter=',')

    return filtered_state_means

filename = './dataset/jena_climate_2009_2016.csv'
hourly_data = resample_data(filename, frequency='1H')
hourly_data = hourly_data.interpolate()

filtered_data = kalman_filter_multivariate(hourly_data)

# Obliczenie MAE dla każdego szeregu czasowego
mae_values = {}
for column in hourly_data.columns:
    mae = mean_absolute_error(hourly_data[column], filtered_data[:, list(hourly_data.columns).index(column)])
    mae_values[column] = mae
    print(f'MAE for {column}: {mae}')

# Zapis MAE do pliku
with open('mae_values.txt', 'w') as file:
    for key, value in mae_values.items():
        file.write(f'{key}: {value}\n')

# Wykres danych oryginalnych i wygładzonych dla temperatury
plt.figure(figsize=(12, 6))
plt.plot(hourly_data.index, hourly_data['T (degC)'], label="Original Data")
plt.plot(hourly_data.index, filtered_data[:, list(hourly_data.columns).index('T (degC)')], label="Kalman Filter Predictions", linestyle="--")
plt.title("Temperature Prediction using Kalman Filter")
plt.legend()
plt.show()
