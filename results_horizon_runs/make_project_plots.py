#!/usr/bin/env python3
"""
Generate plots for the Market Regime Prediction project.

Reads:
  data/daily_returns.csv
  labeled_data/labeled_data.csv
  results_horizon_runs/final_horizon_comparison.csv   (optional)

Saves plots to:
  plots_market_regime/

Run:
  python make_project_plots.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Paths
# ============================================================
DATA_RETURNS = Path("data/daily_returns.csv")
DATA_LABELS = Path("labeled_data/labeled_data.csv")
RESULTS_CSV = Path("results_horizon_runs/final_horizon_comparison.csv")

OUT_DIR = Path("plots_market_regime")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Event windows for highlighting
# ============================================================
EVENT_WINDOWS = [
    {
        "name": "COVID crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "label_y": 0.95,
    },
    {
        "name": "COVID volatility period",
        "start": "2020-02-19",
        "end": "2020-06-30",
        "label_y": 0.85,
    },
    {
        "name": "2025 tariff shock",
        "start": "2025-04-02",
        "end": "2025-05-12",
        "label_y": 0.75,
    },
]


# ============================================================
# Helpers
# ============================================================
def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def add_event_spans(ax, annotate: bool = True) -> None:
    """Shade known market event periods."""
    ymin, ymax = ax.get_ylim()
    for ev in EVENT_WINDOWS:
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"])
        ax.axvspan(start, end, alpha=0.15)
        if annotate:
            y = ymin + (ymax - ymin) * ev["label_y"]
            ax.text(
                start,
                y,
                ev["name"],
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
            )


def add_turbulent_spans(ax, labels: pd.DataFrame, alpha: float = 0.08) -> None:
    """Shade contiguous regions where regime label == 2."""
    if "label" not in labels.columns:
        return

    s = labels["label"].astype(int)
    turbulent = s == 2

    if turbulent.sum() == 0:
        return

    starts = []
    ends = []
    in_region = False
    start = None

    idx = labels.index
    for i, flag in enumerate(turbulent.values):
        if flag and not in_region:
            start = idx[i]
            in_region = True
        if in_region and (not flag or i == len(turbulent) - 1):
            end = idx[i] if flag else idx[i - 1]
            starts.append(start)
            ends.append(end)
            in_region = False

    for st, en in zip(starts, ends):
        ax.axvspan(st, en, alpha=alpha)


def require_files() -> None:
    missing = []
    for p in [DATA_RETURNS, DATA_LABELS]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def load_data():
    require_files()
    returns = pd.read_csv(DATA_RETURNS, index_col=0, parse_dates=True)
    labels = pd.read_csv(DATA_LABELS, index_col=0, parse_dates=True)

    common = returns.index.intersection(labels.index)
    returns = returns.loc[common].sort_index()
    labels = labels.loc[common].sort_index()

    labels = labels.copy()
    labels["market_return"] = returns.mean(axis=1)
    labels["market_cum_return"] = (1.0 + labels["market_return"]).cumprod() - 1.0

    return returns, labels


# ============================================================
# Dataset plots
# ============================================================
def plot_market_cumulative_return(labels: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(labels.index, labels["market_cum_return"])
    add_turbulent_spans(ax, labels)
    add_event_spans(ax)
    ax.set_title("Equal-Weight Market Cumulative Return with Turbulent Regimes and Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    savefig("01_market_cumulative_return_events.png")


def plot_market_features(labels: pd.DataFrame) -> None:
    cols = [
        "market_volatility",
        "avg_correlation",
        "cross_sectional_dispersion",
    ]
    for i, col in enumerate(cols, start=2):
        if col not in labels.columns:
            print(f"Skipping missing column: {col}")
            continue
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(labels.index, labels[col])
        add_turbulent_spans(ax, labels)
        add_event_spans(ax)
        ax.set_title(f"{col.replace('_', ' ').title()} with Turbulent Regimes and Events")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        savefig(f"{i:02d}_{col}_events.png")


def plot_regime_timeline(labels: pd.DataFrame) -> None:
    if "label" not in labels.columns:
        print("Skipping regime timeline: missing label column.")
        return

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.step(labels.index, labels["label"].astype(int), where="post")
    add_event_spans(ax)
    ax.set_title("Market Regime Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Calm", "Normal", "Turbulent"])
    ax.grid(True, alpha=0.3)
    savefig("05_regime_timeline_events.png")


def plot_class_distribution(labels: pd.DataFrame) -> None:
    if "label" not in labels.columns:
        print("Skipping class distribution: missing label column.")
        return

    counts = labels["label"].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Calm", "Normal", "Turbulent"], [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)])
    ax.set_title("Overall Regime Class Distribution")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    savefig("06_regime_class_distribution.png")


def plot_split_class_distribution(labels: pd.DataFrame) -> None:
    if "label" not in labels.columns:
        return

    split_labels = []
    for d in labels.index:
        if d.year <= 2022:
            split_labels.append("Train")
        elif d.year == 2023:
            split_labels.append("Validation")
        else:
            split_labels.append("Test")

    df = pd.DataFrame({"split": split_labels, "label": labels["label"].astype(int).values})
    tab = (
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Train", "Validation", "Test"])
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tab.index))
    width = 0.25

    for offset, lab, name in [(-width, 0, "Calm"), (0, 1, "Normal"), (width, 2, "Turbulent")]:
        ax.bar(x + offset, tab.get(lab, pd.Series(0, index=tab.index)).values, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(tab.index)
    ax.set_title("Regime Class Distribution by Chronological Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig("07_regime_distribution_by_split.png")


# ============================================================
# Model comparison plots
# ============================================================
def load_results() -> pd.DataFrame | None:
    if not RESULTS_CSV.exists():
        print(f"Skipping model comparison plots: {RESULTS_CSV} not found.")
        return None

    df = pd.read_csv(RESULTS_CSV)
    required = {"horizon", "model", "split", "accuracy", "macro_f1", "weighted_f1"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping model comparison plots. Missing columns in {RESULTS_CSV}: {missing}")
        return None

    return df


def plot_macro_f1_by_horizon(df: pd.DataFrame, split: str) -> None:
    sub = df[df["split"].str.lower() == split.lower()].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(index="horizon", columns="model", values="macro_f1", aggfunc="mean").sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", label=model)

    ax.set_title(f"Macro F1 by Horizon ({split.title()})")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(sorted(sub["horizon"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    savefig(f"08_macro_f1_by_horizon_{split.lower()}.png")


def plot_test_macro_f1_bars(df: pd.DataFrame) -> None:
    sub = df[df["split"].str.lower() == "test"].copy()
    if sub.empty:
        return

    # Keep the model order stable and readable.
    models = list(sub["model"].drop_duplicates())
    horizons = sorted(sub["horizon"].unique())
    width = 0.8 / max(len(horizons), 1)
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, h in enumerate(horizons):
        vals = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["horizon"] == h)]
            vals.append(row["macro_f1"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + (i - (len(horizons) - 1) / 2) * width, vals, width, label=f"t+{h}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Test Macro F1 by Model and Horizon")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig("09_test_macro_f1_bar_comparison.png")


def plot_gcn_vs_flattened_gain(df: pd.DataFrame) -> None:
    sub = df[df["split"].str.lower() == "test"].copy()
    if sub.empty:
        return

    # Flexible names in case you renamed models.
    gcn_rows = sub[sub["model"].str.contains("GCN", case=False, na=False)]
    flat_rows = sub[sub["model"].str.contains("Flattened|No-Graph|Flat", case=False, na=False)]

    if gcn_rows.empty or flat_rows.empty:
        print("Skipping GCN-vs-flattened gain plot: could not identify both models.")
        return

    rows = []
    for h in sorted(sub["horizon"].unique()):
        g = gcn_rows[gcn_rows["horizon"] == h]
        f = flat_rows[flat_rows["horizon"] == h]
        if not g.empty and not f.empty:
            rows.append({
                "horizon": h,
                "gcn": g["macro_f1"].max(),
                "flat": f["macro_f1"].max(),
                "gain": g["macro_f1"].max() - f["macro_f1"].max(),
            })

    if not rows:
        return

    gain_df = pd.DataFrame(rows)
    gain_df.to_csv(OUT_DIR / "gcn_vs_flattened_gain.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(gain_df["horizon"].astype(str), gain_df["gain"])
    ax.set_title("GCN Macro-F1 Gain over Fair Flattened No-Graph Baseline")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Macro F1 Gain")
    ax.grid(True, axis="y", alpha=0.3)
    savefig("10_gcn_vs_flattened_gain.png")


def plot_per_class_f1_test(df: pd.DataFrame) -> None:
    sub = df[df["split"].str.lower() == "test"].copy()
    cols = ["f1_calm", "f1_normal", "f1_turbulent"]
    if sub.empty or not all(c in sub.columns for c in cols):
        return

    # Plot only key models if present.
    key_mask = (
        sub["model"].str.contains("GCN", case=False, na=False)
        | sub["model"].str.contains("Flattened|No-Graph|Flat", case=False, na=False)
        | sub["model"].str.contains("Logistic", case=False, na=False)
        | sub["model"].str.contains("SVM", case=False, na=False)
    )
    sub = sub[key_mask].copy()

    for h in sorted(sub["horizon"].unique()):
        cur = sub[sub["horizon"] == h].copy()
        if cur.empty:
            continue

        models = cur["model"].tolist()
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width, cur["f1_calm"].values, width, label="Calm")
        ax.bar(x, cur["f1_normal"].values, width, label="Normal")
        ax.bar(x + width, cur["f1_turbulent"].values, width, label="Turbulent")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(f"Test Per-Class F1 by Model (t+{h})")
        ax.set_xlabel("Model")
        ax.set_ylabel("F1")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        savefig(f"11_per_class_f1_test_h{h}.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Loading dataset...")
    returns, labels = load_data()

    print("Creating dataset plots...")
    plot_market_cumulative_return(labels)
    plot_market_features(labels)
    plot_regime_timeline(labels)
    plot_class_distribution(labels)
    plot_split_class_distribution(labels)

    print("Creating model comparison plots...")
    results = load_results()
    if results is not None:
        plot_macro_f1_by_horizon(results, "val")
        plot_macro_f1_by_horizon(results, "test")
        plot_test_macro_f1_bars(results)
        plot_gcn_vs_flattened_gain(results)
        plot_per_class_f1_test(results)

    print(f"\nDone. All plots saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
