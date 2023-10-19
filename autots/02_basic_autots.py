# also load: _hourly, _monthly, _weekly, _yearly, or _live_daily
from autots import AutoTS, load_daily
import matplotlib.pyplot as plt

# sample datasets can be used in either of the long or wide import shapes
long = False
df = load_daily(long=long)
print(df.head())
df.plot()
plt.show()