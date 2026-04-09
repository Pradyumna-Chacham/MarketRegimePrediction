# Market Regime Prediction with Baselines and GCN

## Overview

This project builds a market regime classification pipeline on a universe of 100 US stocks.

The workflow is:

1. Download adjusted close prices.
2. Clean the price matrix and compute daily returns.
3. Build market regime labels from rolling market statistics.
4. Train baseline models for regime prediction.
5. Train graph neural network models that use stock correlation structure over time.

The project compares two main modeling families:

- `baselines.py` and `baselinetp5.py`: classical machine learning baselines using market-level tabular features.
- `gcn/src` and `gcn/src1`: graph-based models that treat stocks as nodes and rolling correlations as edges.

There are two forecasting settings in the repository:

- Same-time / current-time regime classification:
  `gcn/src`
- Forward regime prediction:
  `baselines.py` for next day, `baselinetp5.py` for `t+5`, and `gcn/src1` for `t+5`

## Problem Setup

Each trading day is assigned one of three market regimes:

- `0`: calm
- `1`: normal
- `2`: turbulent

The label is created from three rolling market features:

- market volatility
- average pairwise stock correlation
- cross-sectional dispersion of stock returns

These are combined into a composite `regime_score`, and train-set quantiles are used to convert the score into 3 classes.

## Environment Setup

This project should be installed from `requirements.txt`.

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are on Windows PowerShell, activate with:

```powershell
venv\Scripts\Activate.ps1
```

## Repository Structure

### Data preparation and analysis

- `scraper.py`
  Downloads market data with `yfinance`, cleans missingness, and writes:
  - `data/raw_prices.csv`
  - `data/raw_missingness.csv`
  - `data/clean_prices.csv`
  - `data/clean_missingness.csv`
  - `data/daily_returns.csv`

- `EDAscript.py`
  Basic exploratory analysis on prices, returns, volatility, and correlation.

- `label_data.py`
  Builds regime labels and saves:
  - `labeled_data/labeled_data.csv`

- `stateval.py`
  Checks label distributions, thresholds, and feature ranges across train, validation, and test splits.

- `turbulent_analysis.py`
  Older analysis script for looking at turbulent periods.

### Baseline models

- `baselines.py`
  Baselines for next-day regime prediction.

- `baselinetp5.py`
  Baselines for `t+5` regime prediction.

- `flattenednode.py`
  Stronger non-graph baseline using per-stock features flattened into one feature vector.

### Graph neural network models

- `gcn/src`
  Same-time GCN + LSTM pipeline.

  Files:
  - `gcn/src/graph_builder.py`
  - `gcn/src/dataset_builder.py`
  - `gcn/src/model.py`
  - `gcn/src/train_gnn.py`

- `gcn/src1`
  `t+5` GCN + LSTM pipeline.

  Files:
  - `gcn/src1/graph_builder_tplus5.py`
  - `gcn/src1/dataset_builder_tplus5.py`
  - `gcn/src1/model_tplus5.py`
  - `gcn/src1/train_gnn_tplus5.py`

## Data Flow

The end-to-end data flow is:

1. `scraper.py`
   Raw prices -> cleaned prices -> daily returns

2. `label_data.py`
   Daily returns -> rolling market features -> composite regime score -> labels

3. Modeling scripts
   - baselines use market-level tabular features from returns and labels
   - GCN models build rolling correlation graphs from returns and learn temporal graph representations

## Train / Validation / Test Split

All major scripts use chronological splits:

- Train: `2017-2022`
- Validation: `2023`
- Test: `2024-2026`

This is important because the project is treating the task as a time-series forecasting / temporal classification problem.

## Sequence of Files to Run

If you want to reproduce the full pipeline from raw data to models, run the scripts in this order.

Before running the scripts, make sure your environment is active and the dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download and prepare the market data

```bash
python scraper.py
```

This creates the cleaned price and return files in `data/`.

### 2. Run exploratory data analysis

```bash
python EDAscript.py
```

This step is optional for training, but useful for sanity checking the data.

### 3. Create the regime labels

```bash
python label_data.py
```

This creates:

- `labeled_data/labeled_data.csv`

### 4. Validate the regime labels

```bash
python stateval.py
```

This step is also optional for training, but recommended to confirm the splits and class balance.

### 5. Run baseline models

Next-day baseline:

```bash
python baselines.py
```

`t+5` baseline:

```bash
python baselinetp5.py
```

Flattened per-stock baseline:

```bash
python flattenednode.py
```

### 6. Train the GCN model for same-time classification

From inside `gcn/src`:

```bash
cd gcn/src
python train_gnn.py
```

### 7. Train the GCN model for `t+5` prediction

From inside `gcn/src1`:

```bash
cd gcn/src1
python train_gnn_tplus5.py
```

## Model Summary

### Baseline models

The baseline scripts use these market-level features:

- average market return
- market volatility
- average correlation
- cross-sectional dispersion

The baseline model set includes:

- Logistic Regression
- K-Means
- Hierarchical Clustering
- SVM
- Gaussian HMM

### GCN models

The graph models treat:

- stocks as nodes
- rolling stock correlations as edges
- node features as short-horizon return and volatility information

For each sample:

- a rolling graph is built from a 30-day window of returns
- a sequence of graphs is formed across time
- a GCN encodes each graph
- an LSTM models the graph sequence over time
- a classifier predicts the regime class

## Notes

- `labeled_data/labeled_data.csv` is the main source-of-truth label file used by the active training scripts.
- The split files in `labeled_data/` appear to be older helper exports and are not the main files used by the current pipelines.
- `baselines.py` is intended for next-day prediction, while `baselinetp5.py` is for `t+5`.
- If model saving fails in `gcn/src/train_gnn.py`, create the output directory first if needed.

## Quick Start

If you only want the shortest useful run order:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scraper.py
python label_data.py
python stateval.py
python baselines.py
python baselinetp5.py
cd gcn/src && python train_gnn.py
cd ../src1 && python train_gnn_tplus5.py
```
