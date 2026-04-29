Here’s a **cleaned, submission-ready README** based on your file (I fixed clarity, consistency, and added a strong results narrative + removed redundancy):

---

# Market Regime Prediction with Graph Neural Networks

## Project Title

**Learning Market Regimes from Time-Evolving Asset Graphs**

---

## Overview

This project builds a machine learning pipeline to predict **market regimes** — *calm (0), normal (1), turbulent (2)* — using a universe of 100 U.S. stocks.

We compare:

* Classical ML baselines
* Tabular deep learning
* Graph Neural Networks (GCN + Attention + LSTM)

The key idea is simple:

> Market behavior is driven not just by individual stocks, but by **how stocks move together (correlation structure)**.

GCNs explicitly model this structure.

---

## Key Contributions

* Multi-horizon prediction (**t+1, t+3, t+5**)
* Graph-based modeling of stock correlations
* Fair comparison against:

  * Classical baselines
  * Non-graph neural models
* Focus on **macro-F1** (handles class imbalance)

---

## Problem Statement

Financial markets exhibit **regimes** characterized by:

* Volatility levels
* Correlation patterns
* Cross-sectional dispersion

Predicting these regimes helps:

* Risk management
* Portfolio allocation
* Strategy switching

---

## Pipeline Overview

```
Raw Prices → Returns → Rolling Features → Regime Score → Labels
                                         ↓
                                 Model Training
                                         ↓
                             Evaluation (Macro-F1)
```

---

## Dataset

* ~100 U.S. stocks
* Daily data
* Time split:

| Split      | Period    |
| ---------- | --------- |
| Train      | 2017–2022 |
| Validation | 2023      |
| Test       | 2024–2026 |

---

## Feature Engineering

Rolling window (30 days):

* Market volatility
* Average correlation
* Cross-sectional dispersion

Normalized using **train-only statistics**

---

## Label Generation

Composite regime score:

```
regime_score = 0.5 * volatility_z
             + 0.3 * correlation_z
             + 0.2 * dispersion_z
```

Thresholds (train-set quantiles):

* Calm: ≤ 25th percentile
* Normal: 25th–80th percentile
* Turbulent: > 80th percentile

---

## Models

### 1. Classical Baselines

* Logistic Regression
* SVM (RBF)
* K-Means
* Hierarchical Clustering
* Gaussian HMM

Uses only **market-level features**

---

### 2. Tabular Deep Learning (Flattened)

* Per-stock features flattened
* Dense + LSTM architecture

Purpose:

> Tests whether deep learning alone helps without graph structure

---

### 3. Graph Neural Network (Main Model)

**Graph Construction**

* Nodes: Stocks
* Edges: Top-K correlations (K=8)
* Dynamic over time

**Architecture**

```
GCN → Node Attention → Temporal Attention → LSTM → Classifier
```

Captures:

* Cross-stock relationships
* Temporal dependencies

---

## Training Details

* Optimizer: Adam
* Loss: Cross-entropy + label smoothing
* Epochs: 20 (early stopping)
* Batch size: 128

---

## Evaluation Metrics

* Accuracy
* **Macro F1 (primary)**
* Weighted F1
* Per-class F1

---

## Results (Test Set)

### Best Model: GCN Attention

| Horizon | Accuracy | Macro F1 |
| ------- | -------- | -------- |
| t+1     | ~0.93    | **0.85** |
| t+3     | ~0.89    | **0.78** |
| t+5     | ~0.87    | **0.69** |

### Observations

* GCN outperforms all baselines across horizons
* Performance degrades as horizon increases (expected)
* Biggest gains are in **minority regimes (calm/turbulent)**

---

## Key Insights

* Correlation structure matters → GCN > tabular models
* Classical models perform surprisingly strong for short horizon
* Flattened deep learning fails to capture minority regimes
* Macro-F1 is critical due to class imbalance

---

## How to Run

### Full Pipeline

```bash
python scraper.py
python label_data.py
python stateval.py
python run_all_horizons_compare.py
```

---

### Individual Models

```bash
# Baselines
python baselinetp_cli.py --horizon 3

# Flattened DL
python flattened_node_fair_cli.py --horizon 3

# GCN
python gcn_attention_cli.py --horizon 3
```

---

## Project Structure

```
data/                  # Raw + processed data
labeled_data/          # Final dataset
stateval_outputs/      # Data validation
results_horizon_runs/  # Model results
gcn/                   # Graph models
```

---

## Known Limitations

* Validation split lacks calm samples
* Class imbalance (turbulent is rare)
* Threshold choice affects label distribution

---

## Future Work

* Transformer-based temporal models
* Learned graph structures (not fixed correlations)
* Add macroeconomic features (VIX, rates)
* Ensemble methods

---

## Conclusion

This project shows:

> **Modeling market structure explicitly (via graphs) improves regime prediction.**

GCNs consistently outperform both classical and deep tabular baselines, especially in detecting difficult regimes.
