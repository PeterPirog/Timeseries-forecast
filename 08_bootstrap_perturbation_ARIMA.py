import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Przykładowe dane
np.random.seed(42)
n_points = 100
x = np.linspace(0, 10, n_points)
y = np.sin(x) + 0.2 * np.random.randn(n_points)
start_parameters = [0.5, 0.5, 0.5]

def bootstrap_arima(y, order=(1, 1, 1), n_iterations=2000):
    # Dopasuj początkowy model ARIMA
    model = ARIMA(y, order=order)
    #model_fit = model.fit(start_params=start_parameters)
    model_fit = model.fit(method='hannan_rissanen')

    # Wygeneruj prognozy i reszty
    predictions = model_fit.predict()
    residuals = y - predictions

    params_list = []

    for _ in range(n_iterations):
        # Bootstrap reszt
        bootstrapped_residuals = np.random.choice(residuals, len(residuals), replace=True)

        # Utwórz nowy szereg czasowy dodając wylosowane reszty do prognoz
        y_new = predictions + bootstrapped_residuals

        # Dopasuj model ARIMA do nowego szeregu czasowego
        model_boot = ARIMA(y_new, order=order)
        model_boot_fit = model_boot.fit()

        # Zapisz parametry dopasowanego modelu
        params_list.append(model_boot_fit.params)

    return predictions, residuals, params_list, model_fit.params


predictions, residuals, params_list, original_params = bootstrap_arima(y)

# Analiza wyników
params_df = pd.DataFrame(params_list)
print(params_df.describe())

# Tworzenie wykresów
plt.figure(figsize=(15, 8))

# Dane oryginalne i prognozy
plt.subplot(2, 1, 1)
plt.plot(x, y, label='Original Data')
plt.plot(x, predictions, color='red', label='Fitted Model')
plt.legend()
plt.title('Original Data & Fitted ARIMA Model')

# Reszty
plt.subplot(2, 1, 2)
plt.plot(x, residuals, color='green')
plt.title('Residuals')
plt.tight_layout()

plt.show()

# Wypisanie parametrów oryginalnego modelu ARIMA
print("\nOriginal ARIMA Model Parameters:")
print(original_params)
