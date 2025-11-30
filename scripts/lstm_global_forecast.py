import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
#  PATHS
# -----------------------------
PRED_DIR = os.path.join("scripts", "LSTM_Predictions")
OUTPUT_DIR = os.path.join("scripts", "LSTM_Global")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
#  METRICS
# -----------------------------
def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def rmse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100


# -----------------------------
#  LOAD ALL LSTM PREDICTIONS
# -----------------------------
all_files = [f for f in os.listdir(PRED_DIR) if f.endswith(".csv")]

if not all_files:
    raise FileNotFoundError("No prediction CSV files found in scripts/LSTM_Predictions/")

dfs = []
for f in all_files:
    path = os.path.join(PRED_DIR, f)
    df = pd.read_csv(path)
    df["Ticker"] = f.replace("predictions_", "").replace(".csv", "")
    df["Date"] = pd.to_datetime(df["Date"])
    dfs.append(df)

# Merge all predictions
full_df = pd.concat(dfs).sort_values("Date").reset_index(drop=True)

# Extract true & predicted values
y_true = full_df["Actual"].values
y_pred = full_df["LSTM_Pred"].values

# -----------------------------
#  GLOBAL METRICS (WEIGHTED)
# -----------------------------
global_mae = mae(y_true, y_pred).mean()
global_rmse = math.sqrt(rmse(y_true, y_pred).mean())
global_mape = mape(y_true, y_pred).mean()

print("==========================================")
print("       GLOBAL WEIGHTED METRICS")
print("==========================================")
print(f"Global MAE:  {global_mae:.6f}")
print(f"Global RMSE: {global_rmse:.6f}")
print(f"Global MAPE: {global_mape:.2f}%")
print("==========================================")

# -----------------------------
#  GLOBAL PLOT
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(full_df["Date"], full_df["Actual"], label="Actual", color="#5e3c99", linewidth=1.2)
plt.plot(full_df["Date"], full_df["LSTM_Pred"], label="LSTM Forecast", color="#f2a104", linewidth=1.2)

plt.title("Global LSTM Forecast vs Actual (All Currencies Combined)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(alpha=0.3)

# Save plot
png_path = os.path.join(OUTPUT_DIR, "global_forecast.png")
svg_path = os.path.join(OUTPUT_DIR, "global_forecast.svg")

plt.savefig(png_path, dpi=300)
plt.savefig(svg_path)
plt.close()

print(f"\nSaved global plot to:\n{png_path}\n{svg_path}")
