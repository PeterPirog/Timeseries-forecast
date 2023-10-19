import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from random import gauss
from random import seed

seed(1)
series = [gauss(0.0, 1.0) for i in range(1000)]
df = pd.DataFrame(series)

plt.figure(figsize=(12, 6))
plt.plot(df)
plt.title("Wygenerowany szereg czasowy")
plt.show()


model = ARIMA(df, order=(0, 0, 2))
model_fit = model.fit()

# Sumaryzujmy dopasowanie modelu
print(model_fit.summary())

# Obejrzyjmy reszty
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

print(residuals.describe())
