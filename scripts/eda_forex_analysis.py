
#  EDA SCRIPT 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. LOAD DATA
# -------------------------------
file_path = r"C:\Users\nikol\Desktop\ThesisNP\data\full_exchange_rates.csv"
df = pd.read_csv(file_path)

# Fix column names (if needed)
df.columns = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# -------------------------------
# 2. AUTO-DETECT ALL CURRENCY PAIRS
# -------------------------------
currency_pairs = sorted(df["Ticker"].unique())
print("Detected currency pairs:", currency_pairs)


# 3. CREATE OUTPUT FOLDERS

base_dir = r"C:\Users\nikol\Desktop\ThesisNP\scripts\EDA_Plots"
png_dir = os.path.join(base_dir, "PNG")
svg_dir = os.path.join(base_dir, "SVG")

os.makedirs(png_dir, exist_ok=True)
os.makedirs(svg_dir, exist_ok=True)

# 4. GENERATE TREND PLOTS

sns.set_style("whitegrid")

for pair in currency_pairs:
    dff = df[df["Ticker"] == pair].copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(dff["Date"], dff["Close"], color="#5A3E9D", linewidth=1.2)  # purple
    
    plt.title(f"{pair} - Long-Term Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    
    plt.tight_layout()
    
    png_path = os.path.join(png_dir, f"Trend_{pair}.png")
    svg_path = os.path.join(svg_dir, f"Trend_{pair}.svg")
    
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    print(f"Saved plots for {pair}")

print("\nAll EDA trend plots completed successfully!")
