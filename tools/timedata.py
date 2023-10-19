import os
import pandas as pd
import matplotlib.pyplot as plt


class ClimateTimeSeries:
    """
    A class to handle and visualize climate time series data.
    """
    COLUMN_DESCRIPTIONS = {
        'Date Time': 'Date-time reference',
        'p (mbar)': 'Pressure in millibars',
        'T (degC)': 'Temperature in Celsius',
        'Tpot (K)': 'Temperature in Kelvin',
        'Tdew (degC)': 'Dew Point in Celsius',
        'rh (%)': 'Relative Humidity percentage',
        'VPmax (mbar)': 'Saturation vapor pressure',
        'VPact (mbar)': 'Vapor pressure',
        'VPdef (mbar)': 'Vapor pressure deficit',
        'sh (g/kg)': 'Specific humidity',
        'H2OC (mmol/mol)': 'Water vapor concentration',
        'rho (g/m ** 3)': 'Density of air',
        'wv (m/s)': 'Wind speed',
        'max. wv (m/s)': 'Maximum wind speed',
        'wd (deg)': 'Wind direction in degrees'
    }

    def __init__(self, resample_frequency='1H', samples=None, selected_parameters=None):
        current_directory = os.path.dirname(__file__)
        self.filename = os.path.join(current_directory, 'jena_climate_2009_2016.csv')
        self.resample_frequency = resample_frequency
        self.samples = samples
        self.selected_parameters = selected_parameters
        self.dataframe = self.prepare_data()
        pd.set_option("display.max_columns", 50)

    def prepare_data(self):
        data = pd.read_csv(self.filename, delimiter=",")
        data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
        data.set_index('Date Time', inplace=True)
        data['wv (m/s)'] = data['wv (m/s)'].clip(lower=0)
        data['max. wv (m/s)'] = data['max. wv (m/s)'].clip(lower=0)

        if self.selected_parameters:
            all_columns = [col for col in self.COLUMN_DESCRIPTIONS.keys() if col != 'Date Time']
            selected_columns = [all_columns[i - 1] for i in self.selected_parameters]
            data = data[selected_columns]

        return self.resample_data(data)

    def resample_data(self, data):
        data_resampled = data.resample(self.resample_frequency).mean()
        return data_resampled.head(self.samples) if self.samples else data_resampled

    def plot_data(self, data=None):
        data = data or self.dataframe
        fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(10, 15))

        for i, column in enumerate(data.columns):
            axes[i].plot(data.index, data[column], label=column)
            axes[i].set_title(column)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def describe_all_columns(self):
        for column, description in self.COLUMN_DESCRIPTIONS.items():
            print(f"{column}: {description}")


if __name__ == "__main__":
    climate_data = ClimateTimeSeries(resample_frequency='1H', samples=None, selected_parameters=[1, 2, 5])
    print(climate_data.dataframe)
    climate_data.plot_data()
