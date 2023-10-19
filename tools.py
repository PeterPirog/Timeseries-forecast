import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesAnalysis:
    COLUMN_DESCRIPTIONS = {
        'Date Time': 'Date-time reference',
        'p (mbar)': 'The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.',
        'T (degC)': 'Temperature in Celsius',
        'Tpot (K)': 'Temperature in Kelvin',
        'Tdew (degC)': 'Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.',
        'rh (%)': 'Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.',
        'VPmax (mbar)': 'Saturation vapor pressure',
        'VPact (mbar)': 'Vapor pressure',
        'VPdef (mbar)': 'Vapor pressure deficit',
        'sh (g/kg)': 'Specific humidity',
        'H2OC (mmol/mol)': 'Water vapor concentration',
        'rho (g/m ** 3)': 'Airtight',
        'wv (m/s)': 'Wind speed',
        'max. wv (m/s)': 'Maximum wind speed',
        'wd (deg)': 'Wind direction in degrees'
    }

    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()
        pd.set_option("display.max_columns", 50)

    def load_data(self):
        data = pd.read_csv(self.filename, delimiter=",")
        data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
        data.set_index('Date Time', inplace=True)
        data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
        data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)
        return data

    def resample_data(self, frequency='H'):
        return self.data.resample(frequency).mean()

    def plot_data(self, data=None):
        if data is None:
            data = self.data

        fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(10, 15))

        for i, column in enumerate(data.columns):
            axes[i].plot(data.index, data[column], label=column)
            axes[i].set_title(column)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def describe_all_columns(self):
        for column, description in self.COLUMN_DESCRIPTIONS.items():
            print(f"{column}:{description}")

# Usage:
if __name__ == "__main__":
    analysis = TimeSeriesAnalysis('./dataset/jena_climate_2009_2016.csv')
    hourly_data = analysis.resample_data(frequency='1H')
    print(hourly_data.head())
    print(hourly_data.describe())
    analysis.plot_data(hourly_data)
    analysis.describe_all_columns()
