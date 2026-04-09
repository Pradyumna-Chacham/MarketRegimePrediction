#BASELINES PREDICTING T+1
# 
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid
from hmmlearn.hmm import GaussianHMM


HORIZON = 1  # predict regime at t+5


# --------------------------------------------------
# Load data
# --------------------------------------------------
returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)
labels_df = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

common_idx = returns.index.intersection(labels_df.index)
returns = returns.loc[common_idx].copy()
labels_df = labels_df.loc[common_idx].copy()

print("Returns shape:", returns.shape)
print("Labels shape:", labels_df.shape)
print("Date range:", returns.index.min(), "→", returns.index.max())


# --------------------------------------------------
# Feature engineering at time t
# --------------------------------------------------
market_return = returns.mean(axis=1)

required_cols = [
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
    "label",
]
missing_cols = [c for c in required_cols if c not in labels_df.columns]
if missing_cols:
    raise ValueError(
        f"Missing columns in labeled_data/labeled_data.csv: {missing_cols}. "
        "Please regenerate labels with label_data.py first."
    )

features = pd.DataFrame({
    "market_return": market_return,
}).join(
    labels_df[[
        "market_volatility",
        "avg_correlation",
        "cross_sectional_dispersion",
        "label",
    ]],
    how="inner",
)

features = features.dropna().copy()

# --------------------------------------------------
# Target at t+5
# --------------------------------------------------
features["future_label"] = features["label"].shift(-HORIZON)

# drop rows near the end that do not have a t+5 target
features = features.dropna(subset=["future_label"]).copy()
features["future_label"] = features["future_label"].astype(int)

print("\nAfter applying t+5 target:")
print("Feature dataframe shape:", features.shape)
print("Feature columns:", list(features.columns))
print("Target horizon:", HORIZON)


# --------------------------------------------------
# Split
# --------------------------------------------------
train_df = features.loc["2017":"2022"].copy()
val_df   = features.loc["2023"].copy()
test_df  = features.loc["2024":"2026"].copy()

feature_cols = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]

X_train = train_df[feature_cols].values
y_train = train_df["future_label"].values

X_val = val_df[feature_cols].values
y_val = val_df["future_label"].values

X_test = test_df[feature_cols].values
y_test = test_df["future_label"].values

print("\nSplit sizes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

print("\nFuture-label class distribution:")
print("Train:\n", pd.Series(y_train).value_counts().sort_index())
print("Val:\n", pd.Series(y_val).value_counts().sort_index())
print("Test:\n", pd.Series(y_test).value_counts().sort_index())


# --------------------------------------------------
# Standardize
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro F1:", round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def build_cluster_label_map(cluster_ids, true_labels):
    mapping = {}
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        mapping[c] = pd.Series(true_labels[mask]).mode()[0]
    return mapping


def apply_cluster_label_map(cluster_ids, mapping):
    return np.array([mapping[c] for c in cluster_ids])


# --------------------------------------------------
# 1. Logistic Regression
# --------------------------------------------------
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

log_reg.fit(X_train_scaled, y_train)

val_pred_lr = log_reg.predict(X_val_scaled)
test_pred_lr = log_reg.predict(X_test_scaled)

evaluate_model("Logistic Regression (t+5) - Validation", y_val, val_pred_lr)
evaluate_model("Logistic Regression (t+5) - Test", y_test, test_pred_lr)


# --------------------------------------------------
# 2. K-Means
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20,
)

train_clusters = kmeans.fit_predict(X_train_scaled)
cluster_map = build_cluster_label_map(train_clusters, y_train)

val_clusters = kmeans.predict(X_val_scaled)
test_clusters = kmeans.predict(X_test_scaled)

val_pred_km = apply_cluster_label_map(val_clusters, cluster_map)
test_pred_km = apply_cluster_label_map(test_clusters, cluster_map)

evaluate_model("K-Means (t+5) - Validation", y_val, val_pred_km)
evaluate_model("K-Means (t+5) - Test", y_test, test_pred_km)


# --------------------------------------------------
# 3. Hierarchical Clustering
# --------------------------------------------------
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward",
)

train_clusters_hc = agg.fit_predict(X_train_scaled)
cluster_map_hc = build_cluster_label_map(train_clusters_hc, y_train)

centroid_model = NearestCentroid()
centroid_model.fit(X_train_scaled, train_clusters_hc)

val_clusters_hc = centroid_model.predict(X_val_scaled)
test_clusters_hc = centroid_model.predict(X_test_scaled)

val_pred_hc = apply_cluster_label_map(val_clusters_hc, cluster_map_hc)
test_pred_hc = apply_cluster_label_map(test_clusters_hc, cluster_map_hc)

evaluate_model("Hierarchical Clustering (t+5) - Validation", y_val, val_pred_hc)
evaluate_model("Hierarchical Clustering (t+5) - Test", y_test, test_pred_hc)


# --------------------------------------------------
# 4. SVM (RBF)
# --------------------------------------------------
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42,
)

svm.fit(X_train_scaled, y_train)

val_pred_svm = svm.predict(X_val_scaled)
test_pred_svm = svm.predict(X_test_scaled)

evaluate_model("SVM (RBF, t+5) - Validation", y_val, val_pred_svm)
evaluate_model("SVM (RBF, t+5) - Test", y_test, test_pred_svm)


# --------------------------------------------------
# 5. Hidden Markov Model
# --------------------------------------------------
hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    min_covar=1e-3,
    random_state=42,
)

hmm.fit(X_train_scaled)


def build_state_map(states, labels):
    mapping = {}
    for s in np.unique(states):
        mask = states == s
        mapping[s] = pd.Series(labels[mask]).mode()[0]
    return mapping


def apply_state_map(states, mapping):
    return np.array([mapping[s] for s in states])


train_states = hmm.predict(X_train_scaled)
state_map = build_state_map(train_states, y_train)

val_states = hmm.predict(X_val_scaled)
test_states = hmm.predict(X_test_scaled)

val_pred_hmm = apply_state_map(val_states, state_map)
test_pred_hmm = apply_state_map(test_states, state_map)

evaluate_model("HMM (t+5) - Validation", y_val, val_pred_hmm)
evaluate_model("HMM (t+5) - Test", y_test, test_pred_hmm)