import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels import api as sm
from sklearn.metrics import mean_squared_error


data = sm.datasets.sunspots.load_pandas().data
ts = pd.Series(
    data["SUNACTIVITY"].values, index=pd.Index(data["YEAR"].astype(int), name="year")
)

ts.plot(title="Sunspots (time series)")
plt.xlabel("Year")
plt.ylabel("Sunspot activity")
plt.show()


n = len(ts)
train_size = int(n * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

print(f"Train length = {len(train)}, Test length = {len(test)}")


order = (2, 1, 2)
model = ARIMA(train, order=order)
fitted = model.fit()
print(fitted.summary())


start_in = train.index[0]
end_in = train.index[-1]
fitted_values = fitted.predict(start=start_in, end=end_in)


start_fc = test.index[0]
end_fc = test.index[-1]
forecast = fitted.predict(start=start_fc, end=end_fc, dynamic=False)


mse = mean_squared_error(test, forecast)
print(f"Test MSE = {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="gray")
plt.plot(fitted_values.index, fitted_values, label="Fitted (train)", color="orange")
plt.plot(forecast.index, forecast, label="Forecast", color="red", marker="o")
plt.legend()
plt.title(f"ARIMA{order} — Forecast vs Actual (Test MSE={mse:.4f})")
plt.show()


future_steps = 5
fc_start = test.index[-1] + 1
fc_end = fc_start + future_steps - 1
future_forecast = fitted.predict(start=fc_start, end=fc_end)
print("Future forecast:\n", future_forecast)
