import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("labeled_data")
PLOT_DIR = Path("plots_market_regime")
PLOT_DIR.mkdir(exist_ok=True)
# returns.to_csv(OUTPUT_DIR / "daily_returns.csv")
# ----------------------------
# Load labeled data
# ----------------------------
df = pd.read_csv(OUTPUT_DIR / "labeled_data.csv", index_col=0, parse_dates=True)

# ----------------------------
# Extract turbulent periods
# ----------------------------
turbulent = df[df["label"] == 2]

print("Total turbulent days:", len(turbulent))

# ----------------------------
# Show first few turbulent dates
# ----------------------------
print("\nSample turbulent dates:")
print(turbulent.head(10))

# ----------------------------
# Group into continuous periods
# ----------------------------
turbulent = turbulent.copy()
turbulent["gap"] = (turbulent.index.to_series().diff().dt.days > 1).cumsum()

periods = turbulent.groupby("gap").agg(
    start=("market_volatility", lambda x: x.index.min()),
    end=("market_volatility", lambda x: x.index.max()),
    length=("market_volatility", "count")
)

print("\nTurbulent periods:")
print(periods)

# ----------------------------
# Plot volatility with turbulence highlighted
# ----------------------------
plt.figure(figsize=(12, 6))

# Plot full volatility
plt.plot(df.index, df["market_volatility"], label="Volatility")

# Highlight turbulent points
plt.scatter(
    turbulent.index,
    turbulent["market_volatility"],
    s=10,
    label="Turbulent",
)

plt.title("Market Volatility with Turbulent Periods Highlighted")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "turbulent_periods.png", dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {PLOT_DIR / 'turbulent_periods.png'}")