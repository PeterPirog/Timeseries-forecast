# https://towardsdatascience.com/multi-seasonal-time-series-decomposition-using-mstl-in-python-136630e67530
# https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.detrend.mstl.MSTL.html

import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.seasonal import MSTL

from tools import TimeSeriesAnalysis

if __name__ == "__main__":
    analysis = TimeSeriesAnalysis('./dataset/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')
    df=df.iloc[:1000]
  # 1. Transformacja Yeo-Johnson
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    df_transformed = transformer.fit_transform(df[['T (degC)']])
    df['T_transformed'] = df_transformed

    # 2. Analiza MSTL na przetransformowanych danych
    res = MSTL(
        endog=df['T_transformed'],
        #periods=(24, 24 * 365),
        periods=(24),
        windows=None,
        lmbda=None,
        iterate=2,
        stl_kwargs={
            'seasonal_deg': 1,
            'robust': True,
        }
    ).fit()

    # 3. Odwrotna transformacja składników
    trend_transformed = transformer.inverse_transform(res.trend.values.reshape(-1, 1)).flatten()
    seasonal_transformed = transformer.inverse_transform(res.seasonal.values.reshape(-1, 1)).flatten()
    resid_transformed = transformer.inverse_transform(res.resid.values.reshape(-1, 1)).flatten()

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(trend_transformed, label="Trend")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(seasonal_transformed, label="Seasonal")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(resid_transformed, label="Residual")
    plt.legend()
    plt.tight_layout()
    plt.show()

  




