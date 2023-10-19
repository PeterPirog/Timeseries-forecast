import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import stats
import numpy as np

# Automatyczne pobieranie danych
url = "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5478/8/1/10/cena_1_m2_powierzchni_uzytkowej_budynku_mieszkalnego_oddanego_do_uzytkowania.csv"
df = pd.read_csv(url, encoding='ISO-8859-2', sep=';', engine='python')

# Utworzenie kolumny 'data' jako połączenie roku i kwartału
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal']*3-2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])

# Sortowanie danych względem daty
df = df.sort_values(by='data')
df.set_index('data', inplace=True)

# Przygotowanie danych dla dekompozycji STL
data_series = df['Wartosc']

# Transformacja Cox-Boxa
data_series_transformed, lambda_value = stats.boxcox(data_series)

# STL Dekompozycja na przekształconych danych
res = STL(
    endog=data_series_transformed,
    period=4,  # Okres sezonowości co 4 próbki (roczna sezonowość)
    robust=True
).fit()

# Odwrotna transformacja Cox-Boxa
if lambda_value == 0:
    df['trend'] = np.exp(res.trend)
else:
    df['trend'] = (lambda_value * res.trend + 1) ** (1 / lambda_value)
df['seasonal'] = data_series / (df['trend'] + res.resid)

# Wyświetlenie wyników
res.plot()
plt.show()

print(df['seasonal'])
