"""
Parameterized classical baselines for market regime prediction.
Runs Logistic Regression, K-Means, Hierarchical clustering, SVM, and HMM
for a chosen prediction horizon.

Example:
    python baselinetp_cli.py --horizon 3 --results_dir results_horizon_runs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon, e.g. 1, 3, 5")
    parser.add_argument("--results_dir", type=str, default="results_horizon_runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


FEATURE_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]


def evaluate_model(name: str, split: str, horizon: int, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    print(f"\n{'=' * 70}")
    print(f"{name} (t+{horizon}) - {split}")
    print(f"{'=' * 70}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro:.4f}")
    print(f"Weighted F1 : {weighted:.4f}")
    print(f"Per-class F1: calm={per[0]:.4f}, normal={per[1]:.4f}, turbulent={per[2]:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0))

    return {
        "horizon": horizon,
        "model": name,
        "split": split,
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "f1_calm": per[0],
        "f1_normal": per[1],
        "f1_turbulent": per[2],
    }


def build_cluster_label_map(cluster_ids: np.ndarray, true_labels: np.ndarray) -> Dict[int, int]:
    mapping = {}
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        mapping[int(c)] = int(pd.Series(true_labels[mask]).mode()[0])
    return mapping


def apply_cluster_label_map(cluster_ids: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(c)] for c in cluster_ids])


def load_features(horizon: int):
    returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)
    labels_df = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

    common_idx = returns.index.intersection(labels_df.index)
    returns = returns.loc[common_idx].copy().sort_index()
    labels_df = labels_df.loc[common_idx].copy().sort_index()

    required_cols = ["market_volatility", "avg_correlation", "cross_sectional_dispersion", "label"]
    missing_cols = [c for c in required_cols if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in labeled_data/labeled_data.csv: {missing_cols}")

    features = pd.DataFrame({"market_return": returns.mean(axis=1)}).join(
        labels_df[["market_volatility", "avg_correlation", "cross_sectional_dispersion", "label"]],
        how="inner",
    )
    features = features.dropna().copy()
    features["future_label"] = features["label"].shift(-horizon)
    features = features.dropna(subset=["future_label"]).copy()
    features["future_label"] = features["future_label"].astype(int)
    return features


def main() -> None:
    args = parse_args()
    horizon = args.horizon
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running classical baselines for horizon={horizon}")
    features = load_features(horizon)

    train_df = features.loc["2017":"2021"].copy()
    val_df = features.loc["2022":"2023"].copy()
    test_df = features.loc["2024":"2026"].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["future_label"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["future_label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["future_label"].values

    print("Split sizes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    rows: List[Dict] = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=args.seed)
    lr.fit(X_train_scaled, y_train)
    rows.append(evaluate_model("Logistic Regression", "Validation", horizon, y_val, lr.predict(X_val_scaled)))
    rows.append(evaluate_model("Logistic Regression", "Test", horizon, y_test, lr.predict(X_test_scaled)))

    # KMeans
    km = KMeans(n_clusters=3, random_state=args.seed, n_init=20)
    train_clusters = km.fit_predict(X_train_scaled)
    cluster_map = build_cluster_label_map(train_clusters, y_train)
    rows.append(evaluate_model("K-Means", "Validation", horizon, y_val, apply_cluster_label_map(km.predict(X_val_scaled), cluster_map)))
    rows.append(evaluate_model("K-Means", "Test", horizon, y_test, apply_cluster_label_map(km.predict(X_test_scaled), cluster_map)))

    # Hierarchical + nearest centroid for val/test assignment
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    train_clusters_hc = agg.fit_predict(X_train_scaled)
    cluster_map_hc = build_cluster_label_map(train_clusters_hc, y_train)
    centroid_model = NearestCentroid()
    centroid_model.fit(X_train_scaled, train_clusters_hc)
    rows.append(evaluate_model("Hierarchical Clustering", "Validation", horizon, y_val, apply_cluster_label_map(centroid_model.predict(X_val_scaled), cluster_map_hc)))
    rows.append(evaluate_model("Hierarchical Clustering", "Test", horizon, y_test, apply_cluster_label_map(centroid_model.predict(X_test_scaled), cluster_map_hc)))

    # SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=args.seed)
    svm.fit(X_train_scaled, y_train)
    rows.append(evaluate_model("SVM RBF", "Validation", horizon, y_val, svm.predict(X_val_scaled)))
    rows.append(evaluate_model("SVM RBF", "Test", horizon, y_test, svm.predict(X_test_scaled)))

    # HMM
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, min_covar=1e-3, random_state=args.seed)
    hmm.fit(X_train_scaled)
    train_states = hmm.predict(X_train_scaled)
    state_map = build_cluster_label_map(train_states, y_train)
    rows.append(evaluate_model("HMM", "Validation", horizon, y_val, apply_cluster_label_map(hmm.predict(X_val_scaled), state_map)))
    rows.append(evaluate_model("HMM", "Test", horizon, y_test, apply_cluster_label_map(hmm.predict(X_test_scaled), state_map)))

    df = pd.DataFrame(rows)
    out_path = out_dir / f"baselines_h{horizon}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
