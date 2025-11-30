import pandas as pd

df = pd.read_csv("data/full_exchange_rates.csv")

# Convert date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Train: μέχρι 2022
train_df = df[df["Date"].dt.year <= 2022]

# Test: μόνο 2023
test_df = df[df["Date"].dt.year == 2023]

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Finished splitting dataset")
