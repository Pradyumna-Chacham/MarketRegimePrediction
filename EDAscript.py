import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Setup output directory
# ----------------------------
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
prices = pd.read_csv("data/clean_prices.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)

# Save basic info
info_df = pd.DataFrame({
    "metric": ["prices_shape", "returns_shape", "start_date", "end_date"],
    "value": [str(prices.shape), str(returns.shape),
              returns.index.min(), returns.index.max()]
})
info_df.to_csv(OUT_DIR / "data_info.csv", index=False)

# ----------------------------
# 1. Missing values
# ----------------------------
missing_pct = returns.isna().mean() * 100
missing_pct.to_csv(OUT_DIR / "missing_pct_per_asset.csv")

missing_summary = missing_pct.describe()
missing_summary.to_csv(OUT_DIR / "missing_pct_summary.csv")

# ----------------------------
# 2. Return statistics
# ----------------------------
return_stats = returns.describe()
return_stats.to_csv(OUT_DIR / "return_stats.csv")

# Flatten returns for histogram (save raw values)
pd.Series(returns.values.flatten()).to_csv(
    OUT_DIR / "return_distribution_values.csv", index=False
)

# ----------------------------
# 3. Rolling market volatility
# ----------------------------
market_return = returns.mean(axis=1)
rolling_vol = market_return.rolling(30).std()

rolling_vol.to_csv(OUT_DIR / "rolling_market_volatility.csv")

# ----------------------------
# 4. Correlation analysis
# ----------------------------
sample_corr = returns.iloc[-30:].corr()
sample_corr.to_csv(OUT_DIR / "sample_correlation_matrix.csv")

corr_values = sample_corr.values.flatten()
pd.Series(corr_values).to_csv(
    OUT_DIR / "sample_correlation_values.csv", index=False
)

# ----------------------------
# 5. Average correlation over time
# ----------------------------
avg_corrs = []
window = 30

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()
    avg_corrs.append(corr.values[np.triu_indices_from(corr, k=1)].mean())

avg_corrs = pd.Series(avg_corrs, index=returns.index[window:])
avg_corrs.to_csv(OUT_DIR / "avg_correlation_over_time.csv")

# ----------------------------
# 6. Volatility distribution
# ----------------------------
rolling_vol_full = returns.std(axis=1).rolling(30).mean()
rolling_vol_full.to_csv(OUT_DIR / "rolling_volatility_full.csv")

# Save distribution values
rolling_vol_full.dropna().to_csv(
    OUT_DIR / "rolling_volatility_values.csv"
)

# Percentiles
p33 = rolling_vol_full.quantile(0.33)
p66 = rolling_vol_full.quantile(0.66)

thresholds_df = pd.DataFrame({
    "percentile": ["33%", "66%"],
    "value": [p33, p66]
})
thresholds_df.to_csv(OUT_DIR / "volatility_thresholds.csv", index=False)

# ----------------------------
# 7. Regime sanity timeline
# ----------------------------
regime_df = pd.DataFrame({
    "volatility": rolling_vol_full,
    "threshold_33": p33,
    "threshold_66": p66
})

regime_df.to_csv(OUT_DIR / "volatility_with_thresholds.csv")

print(f"\nEDA outputs saved to: {OUT_DIR.resolve()}")