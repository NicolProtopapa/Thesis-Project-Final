# LSTM FOREX FORECASTING SCRIPT
# Trains one LSTM per ticker using train.csv and evaluates on test.csv

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
#  CONFIG
# -----------------------------
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

OUTPUT_DIR = os.path.join("scripts", "LSTM_Plots")
PNG_DIR = os.path.join(OUTPUT_DIR, "PNG")
SVG_DIR = os.path.join(OUTPUT_DIR, "SVG")
PRED_DIR = os.path.join("scripts", "LSTM_Predictions")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

WINDOW_SIZE = 30          # days in each input window
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
#  DATASET HELPER
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # (window_size, 1), scalar target
        return self.x[idx], self.y[idx]


def create_sequences(values, window_size):
    """Create (X, y) sequences from 1D array."""
    xs, ys = [], []
    for i in range(len(values) - window_size):
        x = values[i : i + window_size]
        y = values[i + window_size]
        xs.append(x)
        ys.append(y)
    xs = np.array(xs).reshape(-1, window_size, 1)
    ys = np.array(ys).reshape(-1, 1)
    return xs, ys


# -----------------------------
#  LSTM MODEL
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        # take last time step
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out


# -----------------------------
#  METRICS
# -----------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    # avoid division by zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# -----------------------------
#  MAIN LOGIC
# -----------------------------
def main():
    print(f"Using device: {DEVICE}")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Ensure consistent column names
    # We assume columns: Ticker, Date, Open, High, Low, Close, Adj Close, Volume
    train_df.columns = [
        "Ticker",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "AdjClose",
        "Volume",
    ]
    test_df.columns = train_df.columns

    # Parse dates
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    # Find all tickers
    tickers = sorted(train_df["Ticker"].unique())
    print("Found tickers in train.csv:", tickers)

    metrics_rows = []

    for ticker in tickers:
        print("\n======================================")
        print(f"Processing ticker: {ticker}")
        print("======================================")

        train_t = train_df[train_df["Ticker"] == ticker].sort_values("Date")
        test_t = test_df[test_df["Ticker"] == ticker].sort_values("Date")

        if len(train_t) < WINDOW_SIZE + 10 or len(test_t) < WINDOW_SIZE:
            print(f"Skipping {ticker}: not enough data.")
            continue

        # Use Close price
        train_values = train_t["Close"].values.astype(float)
        test_values = test_t["Close"].values.astype(float)

        # Dates for full series
        full_dates = pd.concat([train_t["Date"], test_t["Date"]]).reset_index(drop=True)
        full_values = np.concatenate([train_values, test_values])

        # Scale using only train data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        full_scaled = scaler.transform(full_values.reshape(-1, 1)).flatten()

        # Build train sequences (only on train_scaled)
        X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)

        # Build sequences on full series; then select only targets that fall in the test part
        X_all, y_all = create_sequences(full_scaled, WINDOW_SIZE)
        # indices of each y in the full series
        target_indices = np.arange(WINDOW_SIZE, len(full_scaled))

        # test region starts where train_values end
        train_len = len(train_values)
        test_mask = target_indices >= train_len

        X_test = X_all[test_mask]
        y_test_scaled = y_all[test_mask]
        test_target_indices = target_indices[test_mask]
        test_dates = full_dates.iloc[test_target_indices].values

        # Create loaders
        train_dataset = SequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Model, loss, optimizer
        model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        model.train()
        for epoch in range(EPOCHS):
            epoch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train MSE: {avg_loss:.6f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            preds_scaled = model(X_test_tensor).cpu().numpy().flatten()

        # Invert scaling
        y_test_scaled = y_test_scaled.flatten()
        # reshape for inverse_transform
        y_test_inv = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        preds_inv = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # Metrics
        mae_val = mae(y_test_inv, preds_inv)
        rmse_val = rmse(y_test_inv, preds_inv)
        mape_val = mape(y_test_inv, preds_inv)

        print(f"{ticker} - MAE:  {mae_val:.6f}")
        print(f"{ticker} - RMSE: {rmse_val:.6f}")
        print(f"{ticker} - MAPE: {mape_val:.2f}%")

        metrics_rows.append(
            {
                "Ticker": ticker,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "MAPE": mape_val,
                "TestPoints": len(y_test_inv),
            }
        )

        # Save predictions to CSV
        pred_df = pd.DataFrame(
            {
                "Date": test_dates,
                "Actual": y_test_inv,
                "LSTM_Pred": preds_inv,
            }
        )
        csv_name = f"predictions_{ticker.replace('=','')}.csv"
        pred_path = os.path.join(PRED_DIR, csv_name)
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

        # Plot Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_dates, y_test_inv, label="Actual", color="#5e3c99")   # purple
        ax.plot(test_dates, preds_inv, label="LSTM Forecast", color="#f2a104")  # orange
        ax.set_title(f"{ticker} - LSTM Forecast vs Actual (Test Set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        fig.tight_layout()

        base_name = f"LSTM_{ticker.replace('=','')}_forecast"
        png_path = os.path.join(PNG_DIR, base_name + ".png")
        svg_path = os.path.join(SVG_DIR, base_name + ".svg")
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)
        plt.close(fig)
        print(f"Saved plots to {png_path} and {svg_path}")

    # Save metrics summary
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join("scripts", "LSTM_metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print("\nSaved metrics summary to", metrics_path)
    else:
        print("\nNo tickers were processed (possibly not enough data).")


if __name__ == "__main__":
    main()
