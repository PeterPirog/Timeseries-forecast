import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Wczytanie danych bezpośrednio ze strony internetowej
url = 'https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv'
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')

# Utworzenie kolumny 'data' jako połączenie roku i kwartału
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal']*3-2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])

# Sortowanie danych względem daty
df = df.sort_values(by='data')

# Przygotowanie danych dla Prophet
df_prophet = df[['data', 'Wartosc']]
df_prophet.columns = ['ds', 'y']

# Utworzenie i wytrenowanie modelu
model = Prophet(yearly_seasonality=4, growth='logistic', interval_width=0.95)
df_prophet['cap'] = df_prophet['y'].max() * 1.5  # Wartość górna dla funkcji logistycznej
df_prophet['floor'] = 0  # Wartość dolna dla funkcji logistycznej

model.fit(df_prophet)

# Prognoza do końca 2025 roku
future_dates = model.make_future_dataframe(periods=10, freq='Q')
future_dates['cap'] = df_prophet['cap'].iloc[0]
future_dates['floor'] = 0

forecast = model.predict(future_dates)

# Wizualizacja prognozy
fig = model.plot(forecast)
plt.title('Prognoza ceny za metr kwadratowy')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.show()
