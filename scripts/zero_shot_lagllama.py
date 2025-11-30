

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from lag_llama.model.lag_llama import LagLlama
from lag_llama.model.utils import load_config



# USER SETTINGS


DATA_PATH = r"C:\Users\nikol\Desktop\ThesisNP\data\full_exchange_rates.csv"

OUTPUT_BASE = r"C:\Users\nikol\Desktop\ThesisNP\LagLlama_ZeroShot"
PRED_PATH = os.path.join(OUTPUT_BASE, "Predictions")
PNG_PATH = os.path.join(OUTPUT_BASE, "Plots", "PNG")
SVG_PATH = os.path.join(OUTPUT_BASE, "Plots", "SVG")

MODEL_CFG = "configs/model/lag-llama-small.yaml"   # Adjust path if needed

TICKERS = ["EURGBP=X", "EURJPY=X", "AUDUSD=X", "EURCHF=X", 
           "EURCAD=X", "EURCNY=X", "EURUSD=X"]     # choose whatever you want


# -------------------------------------------------------------
# CREATE FOLDERS
# -------------------------------------------------------------
os.makedirs(PRED_PATH, exist_ok=True)
os.makedirs(PNG_PATH, exist_ok=True)
os.makedirs(SVG_PATH, exist_ok=True)


# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
print("Loading Lag-Llama model...")
config = load_config(MODEL_CFG)
model = LagLlama(config=config)

print("Model loaded!")


# -------------------------------------------------------------
# ZERO-SHOT FORECASTING LOOP
# -------------------------------------------------------------
def zero_shot_forecast(df, ticker):

    print(f"\n===============================")
    print(f"Processing (zero-shot): {ticker}")
    print(f"===============================")

    ts = df[ticker].dropna().values.astype(float)

    # Scaling
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.reshape(-1, 1)).flatten()

    # Prepare input window (model expects tokens)
    context_length = config.context_length
    input_window = ts_scaled[-context_length:]

    # Zero-shot prediction
    pred_scaled = model.forecast(input_window, prediction_length=1)
    pred_value = pred_scaled[0]

    # Inverse scaling
    pred_real = scaler.inverse_transform(np.array(pred_value).reshape(-1, 1))[0][0]

    # Save CSV
    pred_df = pd.DataFrame({
        "ticker": [ticker],
        "prediction": [pred_real]
    })

    out_csv = os.path.join(PRED_PATH, f"zeroshot_{ticker.replace('=','')}.csv")
    pred_df.to_csv(out_csv, index=False)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(ts[-200:], label="Actual (last 200 days)", color="#4C4CFF")
    plt.axhline(pred_real, color="#FF914D", linestyle="--", label=f"Zero-shot forecast: {pred_real:.4f}")

    plt.title(f"Lag-Llama Zero-Shot Forecast — {ticker}")
    plt.xlabel("Days (last 200)")
    plt.ylabel("Exchange Rate")
    plt.legend()

    # Save PNG + SVG
    png_file = os.path.join(PNG_PATH, f"ZeroShot_{ticker.replace('=','')}.png")
    svg_file = os.path.join(SVG_PATH, f"ZeroShot_{ticker.replace('=','')}.svg")

    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(svg_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved:")
    print(f" - {png_file}")
    print(f" - {svg_file}")
    print(f" - {out_csv}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    # Ensure DATE column exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    # Run zero-shot for all tickers
    for t in TICKERS:
        if t not in df.columns:
            print(f"[WARNING] Column {t} not found in dataset — skipping.")
            continue
        zero_shot_forecast(df, t)

    print("\nFINISHED ZERO-SHOT FORECASTS!")
