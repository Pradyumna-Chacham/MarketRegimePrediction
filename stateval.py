import pandas as pd
from pathlib import Path

# ----------------------------
# Setup output directory
# ----------------------------
OUT_DIR = Path("stateval_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load labeled data
# ----------------------------
df = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

info_df = pd.DataFrame({
    "metric": ["data_shape", "start_date", "end_date"],
    "value": [str(df.shape), df.index.min(), df.index.max()]
})
info_df.to_csv(OUT_DIR / "data_info.csv", index=False)

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
missing_df = pd.DataFrame({"missing_column": missing})
missing_df.to_csv(OUT_DIR / "missing_required_columns.csv", index=False)

if missing:
    raise ValueError(f"Missing columns in labeled data: {missing}")

# ----------------------------
# Split
# ----------------------------
train = df.loc["2017":"2022"].copy()
val = df.loc["2023"].copy()
test = df.loc["2024":"2026"].copy()

split_sizes = pd.DataFrame({
    "split": ["train", "val", "test"],
    "rows": [len(train), len(val), len(test)]
})
split_sizes.to_csv(OUT_DIR / "split_sizes.csv", index=False)

# ----------------------------
# TRAIN regime score stats
# ----------------------------
train_regime_stats = train["regime_score"].describe()
train_regime_stats.to_csv(OUT_DIR / "train_regime_score_stats.csv")

p33 = train["regime_score"].quantile(0.33)
p66 = train["regime_score"].quantile(0.66)

thresholds = pd.DataFrame({
    "metric": ["p33", "p66", "threshold_gap"],
    "value": [p33, p66, p66 - p33]
})
thresholds.to_csv(OUT_DIR / "train_regime_score_thresholds.csv", index=False)

# ----------------------------
# Class distribution
# ----------------------------
class_dist = []

for split_name, split_df in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    counts = split_df["label"].value_counts().sort_index()
    total = len(split_df)

    for label, count in counts.items():
        class_dist.append({
            "split": split_name,
            "label": int(label),
            "count": int(count),
            "percent": count / total * 100 if total > 0 else 0,
        })

class_dist_df = pd.DataFrame(class_dist)
class_dist_df.to_csv(OUT_DIR / "class_distribution_by_split.csv", index=False)

# ----------------------------
# Feature sanity checks
# ----------------------------
feature_rows = []

for col in ["market_volatility", "avg_correlation", "cross_sectional_dispersion"]:
    for split_name, split_df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        feature_rows.append({
            "feature": col,
            "split": split_name,
            "min": split_df[col].min(),
            "max": split_df[col].max(),
            "mean": split_df[col].mean(),
            "std": split_df[col].std(),
            "median": split_df[col].median(),
        })

feature_ranges = pd.DataFrame(feature_rows)
feature_ranges.to_csv(OUT_DIR / "feature_ranges_by_split.csv", index=False)

# ----------------------------
# Label sanity check
# ----------------------------
df[["regime_score", "label"]].head(10).to_csv(
    OUT_DIR / "label_sanity_first_10_rows.csv"
)

# ----------------------------
# Full useful export
# ----------------------------
df[[
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
    "regime_score",
    "label",
]].to_csv(OUT_DIR / "labeled_data_core_columns.csv")

print(f"State validation outputs saved to: {OUT_DIR.resolve()}")