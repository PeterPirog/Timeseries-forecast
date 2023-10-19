import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.seasonal import MSTL

from tools import TimeSeriesAnalysis
import matplotlib.pyplot as plt

if __name__ == "__main__":
    analysis = TimeSeriesAnalysis('../tools/jena_climate_2009_2016.csv')
    df = analysis.resample_data(frequency='1H')

    # Przycinanie do pierwszych 1000 wierszy
    df = df.iloc[:20000]
    data=df[['T (degC)']]
    #print(data)
    # Rysowanie wykresu
    data.plot()
    plt.title('Temperatura w Celsjuszach')  # dodane dla dodania tytułu wykresu
    plt.ylabel('Temperatura (°C)')  # dodane dla opisu osi y
    plt.xlabel('Czas')  # dodane dla opisu osi x
    plt.grid(True)  # dodane aby wyświetlić siatkę
    plt.tight_layout()  # dostosowanie wykresu do ramki
    plt.show()  # wyświetlenie wykresu



    # 2. Analiza MSTL na przetransformowanych danych
    res = MSTL(
        endog=data,
        periods=(24,24*365),
        windows=None,
        lmbda=None,
        iterate=2,
        stl_kwargs={
            'seasonal_deg': 1,
            'robust': True,
        }
    ).fit()

    res.plot()
    plt.tight_layout()
    plt.show()

    residuals=res.resid.values
    print(residuals)
    # test wartośći sredniej

    import numpy as np
    from scipy.stats import ttest_1samp

    # Przykładowe dane residuals
    #residuals = np.array([-3.678832, -2.977015, -3.949186])  # Wstaw tutaj twoje dane

    # Przeprowadzenie jednopróbkowego testu t-studenta
    t_stat, p_value = ttest_1samp(residuals, 0)

    # Wydrukowanie wyników
    print(f"Statystyka t: {t_stat}")
    print(f"Wartość p: {p_value}")

    # Sprawdzenie hipotezy zerowej na poziomie istotności 0.05 (95%)
    alpha = 0.05
    if p_value < alpha:
        print("Odrzucamy hipotezę zerową - wartość średnia z residuals różni się od 0.")
    else:
        print(
            "Nie odrzucamy hipotezy zerowej - nie mamy wystarczających dowodów, by stwierdzić, że wartość średnia z residuals różni się od 0.")

    print(np.mean(residuals))
    """


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
    """