from tools.timedata import ClimateTimeSeries
from tools.residual_analyzer import ResidualAnalysis
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import statsmodels.api as sm


# https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html

if __name__ == "__main__":
    climate_data = ClimateTimeSeries(resample_frequency='1H', samples=24*7*180, selected_parameters=[2])
    df = climate_data.dataframe
    exog_vars = climate_data.dataframe.copy()
    print(df)

    # Extract the column from the DataFrame to pass as a Series
    series = df['T (degC)']

    res = STL(
        endog=series,
        period=24,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=True,  # Using robust to handle potential outliers
    ).fit()

    plt.rc('figure',figsize=(10,10))
    plt.rc('font',size=5)
    res.plot()
    plt.show()

    #df['trend']=res.trend
    #df['seasonal']=res.seasonal
    #df['resid'] = res.resid
    #print(df.head())

    # Residual analysis

    res_analysis = ResidualAnalysis(res.resid,exog_vars=exog_vars)
    print(res_analysis.test_mean_residuals())
    print(res_analysis.test_residuals_variance())
    print(res_analysis.test_normality_jarque_bera())
    print(res_analysis.test_residual_autocorrelation(lags=25))
    print(res_analysis.test_residual_independence_with_spearman())






    """
        plt.hist(df['resid'])
    plt.show()

    sm.graphics.tsa.plot_pacf(
        df['resid'],
        lags=40,
        method="yw",
        alpha=0.05,
        use_vlines=True,
        title='Partial Autocorrelation',
        zero=False,
        vlines_kwargs=None,
    )
    plt.show()
    analysis = ClimateTimeSeries('tools/jena_climate_2009_2016.csv')
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
"""