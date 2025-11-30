import pandas as pd

# Load dataset
df = pd.read_csv("data/full_exchange_rates.csv")

# Fix column name (just in case)
df.rename(columns=lambda x: x.strip(), inplace=True)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Choose the pair you want
PAIR = "GBP=X"   # άλλαξέ το!

pair_df = df[df["Ticker"] == PAIR].copy()

# Sort by date
pair_df = pair_df.sort_values("Date")

# Keep only Date + Close
clean_df = pair_df[["Date", "Close"]].reset_index(drop=True)

clean_df.to_csv("data/clean_series.csv", index=False)

print("Saved cleaned time series → data/clean_series.csv")
print(clean_df.head())
