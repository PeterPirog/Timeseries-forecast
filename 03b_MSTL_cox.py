import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL
from tools import ClimateTimeSeries

if __name__ == "__main__":
    analysis = ClimateTimeSeries('tools/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')

    # Przycinanie do pierwszych 20000 wierszy
    df = df.iloc[:20000]

    # Przekształcenie temperatury z °C na Kelwiny
    df['T_Kelvin'] = df['T (degC)'] + 273.15

    # 2. Analiza MSTL z transformacją Box-Coxa
    res = MSTL(
        endog=df['T_Kelvin'],
        periods=(24, 24 * 7),
        lmbda="auto",  # Automatyczna wartość dla transformacji Box-Coxa
        windows=None,
        iterate=2,
        stl_kwargs={
            'seasonal_deg': 1,
            'robust': True,
        }
    ).fit()

    res.plot()
    plt.tight_layout()
    plt.show()
