from tools import TimeSeriesAnalysis
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt


if __name__ == "__main__":
    analysis = TimeSeriesAnalysis('./dataset/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')
    #print(df.head())

    res=STL(
        endog=df['T (degC)'],
        period=24,
        seasonal=7,
        robust=True
    ).fit()

    plt.rc('figure',figsize=(10,10))
    plt.rc('font',size=5)
    res.plot()
    plt.show()

    df['trend']=res.trend
    df['seasonal']=res.seasonal
    print(df.head())
