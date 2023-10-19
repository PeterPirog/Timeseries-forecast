
from tools.timedata import ClimateTimeSeries

if __name__ == "__main__":
    climate_data = ClimateTimeSeries(resample_frequency='1H', samples=7 * 24, selected_parameters=[1,2])
    df = climate_data.dataframe
    print(df)
    climate_data.plot_data()

