import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)
data = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
ts = data["Passengers"]


ts.plot(title="Monthly Air Passengers")
plt.show()


train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]


model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
fitted = model.fit()


forecast = fitted.forecast(len(test))

mse = mean_squared_error(test, forecast)
print(f"Test MSE: {mse:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test", color="gray")
plt.plot(forecast, label="Forecast", color="red")
plt.legend()
plt.title("Holt-Winters Exponential Smoothing (Additive)")
plt.show()
