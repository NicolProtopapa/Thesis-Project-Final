# run_forex_forecast_auto.py
import os
import pandas as pd
import numpy as np
import torch

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

from lag_llama.gluon.estimator import LagLlamaEstimator

# safe-unpickle classes from GluonTS that appear in the checkpoint
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood, DistributionLoss

# -------------------- Config / paths --------------------
DATA_PATH = "lag-llama/data/full_exchange_rates.csv"
CKPT_PATH = "lag-llama-model/lag-llama.ckpt"
OUTPUT_PATH = "data/forex_forecast.csv"

PREDICTION_LENGTH = 30   # number of days to forecast
CONTEXT_LENGTH = 128     # history length to feed model (will be overridden if checkpoint provides)
NUM_SAMPLES = 100        # number of Monte-Carlo samples from forecast distribution

# ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -------------------- Load CSV & preprocess --------------------
df = pd.read_csv(DATA_PATH)
print("Data loaded, shape:", df.shape)

# auto-detect date column (case-insensitive)
date_candidates = [c for c in df.columns if "date" in c.lower()]
if not date_candidates:
    raise ValueError("No date-like column found in CSV (looked for columns containing 'date').")
date_col = date_candidates[0]

# auto-detect first numeric (target) column
numeric_cols = df.select_dtypes(include="number").columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric column found in CSV to forecast.")
value_col = numeric_cols[0]

print(f"Using date column: '{date_col}', value column: '{value_col}'")

# drop duplicates by date (keep first), parse dates, sort
df = df.drop_duplicates(subset=date_col, keep="first").copy()
df["date"] = pd.to_datetime(df[date_col])
df = df.sort_values("date").set_index("date")

# if you want to fill missing dates to a daily index (creates NaNs for missing days)
df = df.asfreq("D")  # change frequency if needed, e.g., "B" for business days

# optionally forward/backfill missing values — comment out if you prefer NaNs
df[value_col] = df[value_col].ffill().bfill()

# target series as numpy
ts = df[value_col].values
start = df.index[0]

# -------------------- Build GluonTS dataset --------------------
dataset = ListDataset([{"start": start, "target": ts}], freq="D")

# -------------------- Load checkpoint (weights_only=False for complex checkpoints) --------------------
print("Loading checkpoint (this may take a moment)...")
ckpt = torch.load(CKPT_PATH, map_location=torch.device("cpu"), weights_only=False)
# many checkpoints store model hyperparams under "hyper_parameters" -> "model_kwargs"
est_args = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})

# fall back to script defaults if checkpoint doesn't include some fields
input_size = est_args.get("input_size", 1)
n_layer = est_args.get("n_layer", 6)
n_embd_per_head = est_args.get("n_embd_per_head", 64)
n_head = est_args.get("n_head", 8)
scaling = est_args.get("scaling", True)
time_feat = est_args.get("time_feat", True)

# If the checkpoint contains prediction_length/context_length, prefer them
prediction_length = est_args.get("prediction_length", PREDICTION_LENGTH)
context_length = est_args.get("context_length", CONTEXT_LENGTH)

# -------------------- Create estimator --------------------
estimator = LagLlamaEstimator(
    ckpt_path=CKPT_PATH,
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=input_size,
    n_layer=n_layer,
    n_embd_per_head=n_embd_per_head,
    n_head=n_head,
    scaling=scaling,
    time_feat=time_feat,
)

# -------------------- Create predictor with safe globals --------------------
allowed_classes = [
    StudentTOutput,
    NegativeLogLikelihood,
    DistributionLoss,
]

print("Creating predictor (using safe_globals for PyTorch 2.6+)...")
with torch.serialization.safe_globals(allowed_classes):
    predictor = estimator.create_predictor(
        transformation=estimator.create_transformation(),
        module=estimator.create_lightning_module()
    )

# -------------------- Make predictions --------------------
print("Running predictions...")
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=NUM_SAMPLES)
forecasts = list(forecast_it)
tss = list(ts_it)

# take the first (and only) forecast
first_forecast = forecasts[0]

# compute mean forecast from samples
mean_pred = np.mean(first_forecast.samples, axis=0)

# build date index for predictions, starting the day after last observed date
last_date = df.index[-1]
pred_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit="D"), periods=prediction_length, freq="D")

pred_df = pd.DataFrame({"date": pred_dates, "mean_forecast": mean_pred})
pred_df.to_csv(OUTPUT_PATH, index=False)
print("Forecast saved to", OUTPUT_PATH)