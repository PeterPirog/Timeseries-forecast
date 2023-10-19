import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL
from statsmodels.graphics.tsaplots import plot_pacf
from tools import ClimateTimeSeries

if __name__ == "__main__":
    analysis = ClimateTimeSeries('tools/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')

    # Przycinanie do pierwszych 20000 wierszy
    df = df.iloc[:50000]


    # Rysowanie PACF
    plot_pacf(df['T (degC)'], lags=24*365)  # `lags` określa liczbę opóźnień, które chcesz uwzględnić
    plt.title('Partial Autocorrelation Function')
    plt.show()
