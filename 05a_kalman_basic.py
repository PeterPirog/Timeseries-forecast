from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resample_data(filename, frequency='H'):
    data = pd.read_csv(filename, delimiter=",")
    data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
    data.set_index('Date Time', inplace=True)
    data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
    data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)
    resampled_data = data.resample(frequency).mean()
    return resampled_data
def kalman_filter_temperature(data):
    noisy_data = data['T (degC)'].values

    # Zainicjuj filtr Kalmana
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Nauka estymacji filtru Kalmana na danych (ucz się macierzy)
    kf = kf.em(noisy_data, n_iter=10)

    # Użyj filtru Kalmana do wygładzenia danych
    (filtered_state_means, filtered_state_covariances) = kf.filter(noisy_data)

    return filtered_state_means, kf

filename = './dataset/jena_climate_2009_2016.csv'
hourly_data = resample_data(filename, frequency='1H')

hourly_data['T (degC)'] = hourly_data['T (degC)'].interpolate()

# Użycie funkcji:
filtered_temperatures, kf_model = kalman_filter_temperature(hourly_data)

# Wyświetlanie macierzy
print("Transition matrix (A):\n", kf_model.transition_matrices)
print("\nObservation matrix (H):\n", kf_model.observation_matrices)
print("\nTransition covariance (Q):\n", kf_model.transition_covariance)
print("\nObservation covariance (R):\n", kf_model.observation_covariance)
print("\nInitial state mean:\n", kf_model.initial_state_mean)
print("\nInitial state covariance:\n", kf_model.initial_state_covariance)

# Ocena błędu dopasowania
mse = np.mean((hourly_data['T (degC)'].values - filtered_temperatures)**2)
print("\nMean Squared Error:", mse)

# Wykres danych oryginalnych i wygładzonych
plt.figure(figsize=(12, 6))
plt.plot(hourly_data.index, hourly_data['T (degC)'], label='Original Data')
plt.plot(hourly_data.index, filtered_temperatures, label='Filtered Data', linestyle='--')
plt.legend()
plt.title("Original vs Filtered Temperature Data")
plt.show()
