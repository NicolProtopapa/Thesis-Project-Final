import pandas as pd
import matplotlib.pyplot as plt

# Paths
history_path = "lag-llama/data/full_exchange_rates.csv"
forecast_path = "data/forex_forecast.csv"

# Load historical data
df = pd.read_csv(history_path)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# Use the same value column you used in forecasting
value_col = "Open"   # change if needed

# Load forecast data
pred_df = pd.read_csv(forecast_path)
pred_df["date"] = pd.to_datetime(pred_df["date"])

# Plot
plt.figure(figsize=(12, 5))

# Historical
plt.plot(df.index, df[value_col], label="Historical")

# Forecast
plt.plot(pred_df["date"], pred_df["mean_forecast"], label="Forecast")

plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.title("Historical Data + Lag-Llama Forecast")

plt.tight_layout()
plt.show()