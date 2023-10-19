
from tools.timedata import ClimateTimeSeries

if __name__ == "__main__":
    climat_set = ClimateTimeSeries(resample_frequency='1H', samples=7*24,selected_parameters=[1])
    df = climat_set.dataframe
    print(df)

