"""
Index	Features	Format	Description
1	Date Time	01.01.2009 00:10:00	Date-time reference
2	p (mbar)	996.52	The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.
3	T (degC)	-8.02	Temperature in Celsius
4	Tpot (K)	265.4	Temperature in Kelvin
5	Tdew (degC)	-8.9	Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.
6	rh (%)	93.3	Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.
7	VPmax (mbar)	3.33	Saturation vapor pressure
8	VPact (mbar)	3.11	Vapor pressure
9	VPdef (mbar)	0.22	Vapor pressure deficit
10	sh (g/kg)	1.94	Specific humidity
11	H2OC (mmol/mol)	3.12	Water vapor concentration
12	rho (g/m ** 3)	1307.75	Airtight
13	wv (m/s)	1.03	Wind speed
14	max. wv (m/s)	1.75	Maximum wind speed
15	wd (deg)	152.3	Wind direction in degrees
"""

import pandas as pd
pd.set_option("display.max_columns", 50)
from datetime import datetime
import matplotlib.pyplot as plt


def resample_data(filename, frequency='H'):
    # Wczytanie danych z pliku CSV z odpowiednim formatowaniem kolumny "Date Time"
    data = pd.read_csv(filename, delimiter=",")

    # Konwersja kolumny "Date Time" na typ datetime i ustawienie jej jako indeks
    data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
    data.set_index('Date Time', inplace=True)

    # Modyfikacja wartości w kolumnach, aby były nie mniejsze niż 0
    data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
    data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)

    # Próbkuje dane z określoną częstotliwością i oblicza średnią wartość dla każdej kolumny
    resampled_data = data.resample(frequency).mean()

    return resampled_data


def plot_data(data):
    fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(10, 15))

    for i, column in enumerate(data.columns):
        axes[i].plot(data.index, data[column], label=column)
        axes[i].set_title(column)
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Użycie funkcji:
filename = './dataset/jena_climate_2009_2016.csv'
hourly_data = resample_data(filename,frequency='1H')
print(hourly_data.head())
print(hourly_data.describe())
plot_data(hourly_data)