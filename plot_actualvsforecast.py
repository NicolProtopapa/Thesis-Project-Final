import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood, DistributionLoss

# -------------------- Config --------------------
DATA_PATH = "lag-llama/data/full_exchange_rates.csv"
CKPT_PATH = "lag-llama-model/lag-llama.ckpt"

CUTOFF_DATE = pd.to_datetime("2023-07-01")
FORECAST_END_DATE = pd.to_datetime("2023-07-28")
NUM_SAMPLES = 100

OUTPUT_METRICS = "fx_forecast_metrics.csv"
OUTPUT_PLOTS_DIR = "fx_forecast_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# -------------------- Load CSV --------------------
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# -------------------- Load checkpoint --------------------
ckpt = torch.load(CKPT_PATH, map_location=torch.device("cpu"), weights_only=False)
est_args = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})

input_size = est_args.get("input_size", 1)
n_layer = est_args.get("n_layer", 6)
n_embd_per_head = est_args.get("n_embd_per_head", 64)
n_head = est_args.get("n_head", 8)
scaling = est_args.get("scaling", True)
time_feat = est_args.get("time_feat", True)
context_length = est_args.get("context_length", 128)

estimator = LagLlamaEstimator(
    ckpt_path=CKPT_PATH,
    prediction_length=(FORECAST_END_DATE - CUTOFF_DATE).days,
    context_length=context_length,
    input_size=input_size,
    n_layer=n_layer,
    n_embd_per_head=n_embd_per_head,
    n_head=n_head,
    scaling=scaling,
    time_feat=time_feat,
)

allowed_classes = [StudentTOutput, NegativeLogLikelihood, DistributionLoss]
with torch.serialization.safe_globals(allowed_classes):
    predictor = estimator.create_predictor(
        transformation=estimator.create_transformation(),
        module=estimator.create_lightning_module()
    )

# -------------------- Loop over FX tickers --------------------
metrics_list = []

for ticker in df['Ticker'].unique():
    print(f"\nProcessing FX: {ticker}")
    
    fx_df = df[df['Ticker'] == ticker].sort_values('date').set_index('date')
    fx_df = fx_df.asfreq("D").ffill().bfill()  # fill missing dates
    
    # Slice training data
    train_df = fx_df[fx_df.index <= CUTOFF_DATE]
    ts_train = train_df['Close'].values
    start_train = train_df.index[0]

    # Build GluonTS dataset
    dataset = ListDataset([{"start": start_train, "target": ts_train}], freq="D")

    # Make forecast
    forecast_it, _ = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=NUM_SAMPLES)
    forecast = list(forecast_it)[0]
    mean_forecast = np.mean(forecast.samples, axis=0)

    # Forecast dates
    pred_dates = pd.date_range(start=CUTOFF_DATE + pd.Timedelta(1, "D"), end=FORECAST_END_DATE, freq="D")
    forecast_df = pd.DataFrame({"date": pred_dates, "forecast": mean_forecast})

    # Actual values
    actual_df = fx_df.loc[pred_dates, 'Close'].reset_index()
    actual_df.columns = ["date", "actual"]

    # Merge forecast & actual
    compare_df = pd.merge(forecast_df, actual_df, on="date")

    # Compute metrics
    rmse = np.sqrt(np.mean((compare_df["forecast"] - compare_df["actual"])**2))
    mae = np.mean(np.abs(compare_df["forecast"] - compare_df["actual"]))
    mape = np.mean(np.abs((compare_df["forecast"] - compare_df["actual"]) / compare_df["actual"])) * 100

    metrics_list.append({"FX": ticker, "RMSE": rmse, "MAE": mae, "MAPE": mape})

    # -------------------- Plot --------------------
    plt.figure(figsize=(12,6))

    # Historical
    plt.plot(train_df.index, train_df['Close'], label="Historical", marker='o')

    # Forecast mean
    plt.plot(compare_df["date"], compare_df["forecast"], label="Forecast", marker='x', linestyle='--', color='orange')

    # Actual values
    plt.plot(compare_df["date"], compare_df["actual"], label="Actual", marker='s', linestyle='-', color='green')

    # 95% CI
    lower = np.percentile(forecast.samples, 2.5, axis=0)
    upper = np.percentile(forecast.samples, 97.5, axis=0)
    plt.fill_between(compare_df["date"], lower, upper, color='orange', alpha=0.2, label="95% CI")

    # Highlight points outside 95% CI
    outside_idx = (compare_df["actual"] < lower) | (compare_df["actual"] > upper)
    plt.scatter(compare_df["date"][outside_idx], compare_df["actual"][outside_idx],
                color='red', s=50, zorder=5, label="Outside 95% CI")

    # Forecast start line
    plt.axvline(CUTOFF_DATE, color='gray', linestyle=':', label="Forecast Start")

    # RMSE & MAPE text
    plt.text(
        0.98, 0.02, f"RMSE: {rmse:.4f}\nMAPE: {mape:.2f}%",
        transform=plt.gca().transAxes, fontsize=12, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title(f"Lag-Llama Forecast vs Actual for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_PLOTS_DIR, f"{ticker}_forecast.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved: {plot_file}")

# -------------------- Save metrics --------------------
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print(f"\nMetrics saved to {OUTPUT_METRICS}")