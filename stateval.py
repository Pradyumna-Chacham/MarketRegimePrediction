import pandas as pd

# ----------------------------
# Load labeled data (SOURCE OF TRUTH)
# ----------------------------
df = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

print("Data shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())

# ----------------------------
# Check required columns
# ----------------------------
required_cols = [
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
    "regime_score",
    "label",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in labeled data: {missing}")

# ----------------------------
# Split
# ----------------------------
train = df.loc["2017":"2022"].copy()
val   = df.loc["2023"].copy()
test  = df.loc["2024":"2026"].copy()

print("\nSplit sizes:")
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

# ----------------------------
# TRAIN regime score stats
# ----------------------------
print("\nTRAIN regime_score stats:")
print(train["regime_score"].describe())

# Percentiles (used for labeling)
p33 = train["regime_score"].quantile(0.33)
p66 = train["regime_score"].quantile(0.66)

print("\nTRAIN thresholds (regime_score):")
print("33%:", p33)
print("66%:", p66)

print("\nThreshold gap:", p66 - p33)

# ----------------------------
# Class distribution
# ----------------------------
print("\nClass counts (TRAIN):")
print(train["label"].value_counts().sort_index())

print("\nClass counts (VAL):")
print(val["label"].value_counts().sort_index())

print("\nClass counts (TEST):")
print(test["label"].value_counts().sort_index())

# ----------------------------
# Feature sanity checks
# ----------------------------
print("\nFeature ranges:")

for col in ["market_volatility", "avg_correlation", "cross_sectional_dispersion"]:
    print(f"\n{col}:")
    print("Train min/max:", train[col].min(), "/", train[col].max())
    print("Val   min/max:", val[col].min(), "/", val[col].max())
    print("Test  min/max:", test[col].min(), "/", test[col].max())

# ----------------------------
# Optional: check label consistency
# ----------------------------
print("\nLabel sanity check (first 10 rows):")
print(df[["regime_score", "label"]].head(10))