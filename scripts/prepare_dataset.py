import os
import pandas as pd

# Path from datasets
DATA_DIR = "Exchange_Rates"

all_rows = []

# Κάνουμε loop σε όλα τα αρχεία CSV μέσα στο Exchange_Rates
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        filepath = os.path.join(DATA_DIR, file)
        print("Loading:", filepath)
        df = pd.read_csv(filepath)
        all_rows.append(df)

# Ενώνουμε όλα τα CSV σε ένα DataFrame
full_df = pd.concat(all_rows, ignore_index=True)

# Αποθηκεύουμε ενιαίο dataset στο root του project
full_df.to_csv("../full_exchange_rates.csv", index=False)

print("Created full_exchange_rates.csv")
