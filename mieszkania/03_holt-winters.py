import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Wczytanie danych bezpośrednio ze strony internetowej
url = 'https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv'
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')

# Utworzenie kolumny 'data' jako połączenie roku i kwartału
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal']*3-2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])

# Sortowanie danych względem daty
df = df.sort_values(by='data')

# Przygotowanie danych dla modelu
data_series = df.set_index('data')['Wartosc']
data_series.index.freq = "QS-OCT"

# Ustawienie najlepszych parametrów
best_params = {
    'trend': 'add',
    'damped_trend': False,
    'seasonal': 'add',
    'initialization_method': 'estimated',
    'use_boxcox': 0.2821040247066575,
    'seasonal_periods': 4
}

# Utworzenie i wytrenowanie modelu Holta-Wintersa
model = ExponentialSmoothing(data_series, **best_params).fit()

# Prognoza na 10 kolejnych kwartałów
forecast = model.forecast(10)

# Wizualizacja prognozy
plt.figure(figsize=(12,6))
plt.plot(data_series.index, data_series.values, label='Historical')
plt.plot(forecast.index, forecast.values, color='red', label='Forecast')
plt.title('Prognoza ceny za metr kwadratowy przy użyciu modelu Holta-Wintersa')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.show()

current_val=data_series.values[-1]

# Wizualizacja prognozy
plt.figure(figsize=(12,6))
plt.plot(data_series.index, 100*data_series.values/current_val, label='Historical')
plt.plot(forecast.index, 100*forecast.values/current_val, color='red', label='Forecast')
plt.title('Prognoza ceny za metr kwadratowy przy użyciu modelu Holta-Wintersa')
plt.xlabel('Data')
plt.ylabel('Cena %')
plt.legend()
plt.grid(True)
plt.show()

print("Date\t\tPercentage Change")
print("-" * 40)

for date, forecasted_val in zip(forecast.index, forecast.values):
    percentage_change = (forecasted_val - current_val) / current_val * 100
    print(f"{date.date()}\t{percentage_change:.2f}%")