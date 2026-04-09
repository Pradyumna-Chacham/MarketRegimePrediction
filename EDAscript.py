import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
prices = pd.read_csv("data/clean_prices.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)

print("Prices shape:", prices.shape)
print("Returns shape:", returns.shape)
print("Date range:", returns.index.min(), "→", returns.index.max())

# ----------------------------
# 1. Missing values check
# ----------------------------
missing_pct = returns.isna().mean() * 100
print("\nMissing % per asset:")
print(missing_pct.describe())

# ----------------------------
# 2. Return distribution
# ----------------------------
plt.figure()
plt.hist(returns.values.flatten(), bins=100)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.show()

# Check extreme values
print("\nReturn stats:")
print(returns.describe())

# ----------------------------
# 3. Rolling market volatility
# ----------------------------
# Market proxy: average return across assets
market_return = returns.mean(axis=1)

rolling_vol = market_return.rolling(30).std()

plt.figure()
plt.plot(rolling_vol)
plt.title("30-Day Rolling Market Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

# ----------------------------
# 4. Correlation analysis
# ----------------------------
# Sample one window
sample_corr = returns.iloc[-30:].corr()

plt.figure()
plt.imshow(sample_corr, aspect='auto')
plt.colorbar()
plt.title("Sample Correlation Heatmap (Last 30 Days)")
plt.show()

# Histogram of correlations
corr_values = sample_corr.values.flatten()

plt.figure()
plt.hist(corr_values, bins=50)
plt.title("Correlation Distribution (Sample Window)")
plt.xlabel("Correlation")
plt.ylabel("Frequency")
plt.show()

# ----------------------------
# 5. Average correlation over time
# ----------------------------
avg_corrs = []

window = 30
for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()
    avg_corrs.append(corr.values[np.triu_indices_from(corr, k=1)].mean())

avg_corrs = pd.Series(avg_corrs, index=returns.index[window:])

plt.figure()
plt.plot(avg_corrs)
plt.title("Average Pairwise Correlation Over Time")
plt.xlabel("Date")
plt.ylabel("Average Correlation")
plt.show()

# ----------------------------
# 6. Volatility distribution (for labels)
# ----------------------------
rolling_vol_full = returns.std(axis=1).rolling(30).mean()

plt.figure()
plt.hist(rolling_vol_full.dropna(), bins=50)
plt.title("Distribution of Rolling Volatility")
plt.xlabel("Volatility")
plt.ylabel("Frequency")
plt.show()

# Percentiles (IMPORTANT for labels later)
p33 = rolling_vol_full.quantile(0.33)
p66 = rolling_vol_full.quantile(0.66)

print("\nVolatility thresholds (FULL DATA — DO NOT USE FOR TRAINING):")
print("33%:", p33)
print("66%:", p66)

# ----------------------------
# 7. Quick regime sanity check
# ----------------------------
plt.figure()
plt.plot(rolling_vol_full, label="Volatility")
plt.axhline(p33, linestyle="--", label="33%")
plt.axhline(p66, linestyle="--", label="66%")
plt.legend()
plt.title("Volatility with Regime Thresholds (Sanity Check)")
plt.show()