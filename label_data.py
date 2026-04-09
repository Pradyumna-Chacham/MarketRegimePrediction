import numpy as np
import pandas as pd

# ----------------------------
# Load returns
# ----------------------------
returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)

# ----------------------------
# Feature 1: market volatility
# ----------------------------
market_return = returns.mean(axis=1)
market_volatility = market_return.rolling(30).std()

# ----------------------------
# Feature 2: cross-sectional dispersion
# ----------------------------
cross_sectional_dispersion = returns.std(axis=1).rolling(30).mean()

# ----------------------------
# Feature 3: average pairwise correlation
# ----------------------------
window = 30
avg_corr_values = []
avg_corr_index = []

for i in range(window - 1, len(returns)):
    window_returns = returns.iloc[i - window + 1:i + 1]
    corr = window_returns.corr()
    upper = corr.values[np.triu_indices_from(corr, k=1)]
    avg_corr_values.append(np.mean(upper))
    avg_corr_index.append(returns.index[i])

avg_correlation = pd.Series(avg_corr_values, index=avg_corr_index, name="avg_correlation")

# ----------------------------
# Combine features
# ----------------------------
df = pd.DataFrame({
    "market_volatility": market_volatility,
    "cross_sectional_dispersion": cross_sectional_dispersion,
}).join(avg_correlation, how="inner")

df = df.dropna().copy()

# ----------------------------
# Split train only for normalization
# ----------------------------
train_df = df.loc["2017":"2022"].copy()

# z-score using TRAIN ONLY
for col in ["market_volatility", "avg_correlation", "cross_sectional_dispersion"]:
    mean_train = train_df[col].mean()
    std_train = train_df[col].std()
    df[col + "_z"] = (df[col] - mean_train) / (std_train + 1e-8)

# ----------------------------
# Composite regime score
# ----------------------------
df["regime_score"] = (
    0.5 * df["market_volatility_z"] +
    0.3 * df["avg_correlation_z"] +
    0.2 * df["cross_sectional_dispersion_z"]
)

# ----------------------------
# Thresholds from TRAIN ONLY
# ----------------------------
train_score = df.loc["2017":"2022", "regime_score"]

p33 = train_score.quantile(0.33)
p66 = train_score.quantile(0.66)

print("Train-only regime score thresholds:")
print("33%:", p33)
print("66%:", p66)

# ----------------------------
# Assign labels
# ----------------------------
def assign_label(score):
    if score < p33:
        return 0   # calm
    elif score <= p66:
        return 1   # normal
    else:
        return 2   # turbulent

df["label"] = df["regime_score"].apply(assign_label)

# ----------------------------
# Save
# ----------------------------
df.to_csv("labeled_data/labeled_data.csv")

print("\nClass counts overall:")
print(df["label"].value_counts().sort_index())

print("\nTrain distribution:")
print(df.loc["2017":"2022", "label"].value_counts().sort_index())

print("\nVal distribution:")
print(df.loc["2023", "label"].value_counts().sort_index())

print("\nTest distribution:")
print(df.loc["2024":"2026", "label"].value_counts().sort_index())