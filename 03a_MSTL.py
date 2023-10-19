import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.seasonal import MSTL

from tools import ClimateTimeSeries

if __name__ == "__main__":
    analysis = ClimateTimeSeries('tools/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')

    # Przycinanie do pierwszych 1000 wierszy
    df = df.iloc[:20000]

    # 1. Transformacja Yeo-Johnson
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    df_transformed = transformer.fit_transform(df[['T (degC)']])
    #df['T_transformed'] = df_transformed
    df['T_transformed'] = df[['T (degC)']]

    # 2. Analiza MSTL na przetransformowanych danych
    res = MSTL(
        endog=df['T_transformed'],
        periods=(24, 24 * 365),
        windows=None,
        lmbda=None,
        iterate=2,
        stl_kwargs={
            'seasonal_deg': 1,
            'robust': True,
        }
    ).fit()

    # 3. Odwrotna transformacja składników
    #res.trend = transformer.inverse_transform(res.trend.values.reshape(-1, 1)).flatten()
    #res.seasonal = transformer.inverse_transform(res.seasonal.values.reshape(-1, 1)).flatten()
    #res.resid = transformer.inverse_transform(res.resid.values.reshape(-1, 1)).flatten()

    res.plot()
    plt.tight_layout()
    plt.show()
