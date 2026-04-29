# Market Regime Prediction with Graph Neural Networks

## Project Title
**Learning Market Regimes from Time-Evolving Asset Graphs**


## Overview

This project develops a machine learning pipeline for predicting **market regime transitions** (calm, normal, turbulent) using a universe of 100 US stocks. It compares classical machine learning baselines against graph neural network (GCN) models that explicitly leverage the dynamic correlation structure between stocks.

### Problem Statement
Financial markets exhibit distinct regimes characterized by different volatility and correlation dynamics. Accurate regime prediction has applications in portfolio allocation, risk management, and trading strategy design. This project tackles the problem of **classifying daily market regimes** using both tabular features and learned graph representations.

### Why This Matters
- **Risk Management**: Early detection of turbulent regimes enables portfolio hedging
- **Strategy Adaptation**: Regime-aware trading strategies can adjust parameters dynamically
- **Correlation Dynamics**: The GCN approach captures how stock correlations evolve during market stress

---

## Features

- ✅ **Automated Data Pipeline**: Yahoo Finance scraping with missingness cleaning
- ✅ **Multi-Horizon Forecasting**: Predict market regimes 1, 3, or 5 days ahead
- ✅ **Diverse Model Baselines**: Logistic Regression, K-Means, SVM, Hierarchical Clustering, Gaussian HMM
- ✅ **Graph Neural Networks**: Temporal GCN + LSTM architecture capturing correlation dynamics
- ✅ **Tabular Deep Learning**: Flattened baseline for fair comparison with neural network-based approaches
- ✅ **Comprehensive Evaluation**: Metrics across train/validation/test splits with macro-F1 focus
- ✅ **Parameterized CLI**: Easy experiment management across different horizons
- ✅ **Visualization Tools**: Regime timeline plots with market event annotations

---

## Project Structure

### Core Scripts (Root Level)

| Script | Purpose | Output |
|--------|---------|--------|
| `scraper.py` | Download and clean price data from Yahoo Finance | `data/clean_prices.csv`, `data/daily_returns.csv` |
| `label_data.py` | Compute market features and regime labels | `labeled_data/labeled_data.csv` |
| `stateval.py` | Validate label distributions and feature ranges | `stateval_outputs/*.csv` |
| `EDAscript.py` | Exploratory data analysis (optional) | `eda_outputs/*.csv` |
| `turbulent_analysis.py` | Analyze turbulent market periods | Console output + visualization |
| `make_project_plots.py` | Generate regime timeline visualizations | `plots_market_regime/*.png` |

### Modeling Scripts (CLI Interface)

| Script | Models | Horizon | Output |
|--------|--------|---------|--------|
| `baselinetp_cli.py` | LR, K-Means, SVM, HC, HMM (classical) | Configurable | `results_horizon_runs/baselines_h{1,3,5}.csv` |
| `flattened_node_fair_cli.py` | Deep learning (flattened features, no graph) | Configurable | `results_horizon_runs/flattened_fair_h{1,3,5}.csv` |
| `gcn_attention_cli.py` | GCN + Temporal Attention + LSTM | Configurable | `results_horizon_runs/gcn_attention_h{1,3,5}.csv` |
| `run_all_horizons_compare.py` | Meta-script: runs all models across horizons | 1, 3, 5 (default) | Consolidated comparison CSVs |

### Directory Structure

```
MarketRegimePrediction/
├── data/                              # Raw and cleaned price/return data
│   ├── raw_prices.csv
│   ├── clean_prices.csv
│   ├── daily_returns.csv
│   └── ...
├── labeled_data/                      # Market regime labels and splits
│   ├── labeled_data.csv               # Main label file (source of truth)
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── eda_outputs/                       # Exploratory analysis outputs
│   ├── return_stats.csv
│   ├── volatility_thresholds.csv
│   └── ...
├── stateval_outputs/                  # Label validation outputs
│   ├── class_distribution_by_split.csv
│   ├── train_regime_score_stats.csv
│   └── ...
├── results_horizon_runs/              # Model training results
│   ├── baselines_h1.csv, h3.csv, h5.csv
│   ├── flattened_fair_h1.csv, h3.csv, h5.csv
│   ├── gcn_attention_h1.csv, h3.csv, h5.csv
│   ├── final_horizon_comparison.csv
│   ├── final_test_only_ranking.csv
│   └── final_test_macro_f1_pivot.csv
├── plots_market_regime/               # Visualization outputs
│   └── *.png files
├── cache_gcn/                         # GCN precomputed graphs and splits
├── cache_flat_fair/                   # Flattened model cache
├── gcn/src/                           # Same-time GCN models
│   ├── graph_builder.py
│   ├── dataset_builder.py
│   ├── model.py
│   └── train_gnn.py
├── gcn/src1/                          # Forward (t+5) GCN models
│   ├── graph_builder_tplus5.py
│   ├── dataset_builder_tplus5.py
│   ├── model_tplus5.py
│   └── train_gnn_tplus5.py
├── requirements.txt
├── README.md
└── [core scripts mentioned above]
```

---

## Installation

### Prerequisites
- Python 3.8+
- macOS, Linux, or Windows (with PowerShell)

### Step-by-Step Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd MarketRegimePrediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verification
To verify the installation, run:
```bash
python -c "import torch; import pandas; print('Installation successful!')"
```

---

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline from raw data to model evaluation:

```bash
# 1. Data preparation
python scraper.py              # Download and clean prices
python label_data.py           # Create regime labels
python stateval.py             # Validate labels

# 2. Train all models across horizons 1, 3, 5
python run_all_horizons_compare.py

# Results are saved in results_horizon_runs/
```

### Individual Model Training

#### Train Classical Baselines
```bash
# Next-day prediction (horizon 1)
python baselinetp_cli.py --horizon 1 --results_dir results_horizon_runs

# 5-day-ahead prediction (horizon 5)
python baselinetp_cli.py --horizon 5 --results_dir results_horizon_runs
```

#### Train Flattened Tabular Deep Learning Baseline
```bash
python flattened_node_fair_cli.py --horizon 3 --results_dir results_horizon_runs
```

#### Train Graph Neural Network (GCN + Temporal Attention)
```bash
python gcn_attention_cli.py --horizon 3 --results_dir results_horizon_runs --seed 42
```

### Data Preparation Only

If you only need prepared data without training models:

```bash
python scraper.py              # Creates data/daily_returns.csv
python label_data.py           # Creates labeled_data/labeled_data.csv
python stateval.py             # Optional: validate data
```

### Exploratory Data Analysis (Optional)

```bash
python EDAscript.py                    # Generate EDA statistics
python turbulent_analysis.py           # Analyze turbulent periods
python make_project_plots.py           # Create timeline visualizations
```

### Command-Line Arguments Reference

#### `baselinetp_cli.py` / `flattened_node_fair_cli.py` / `gcn_attention_cli.py`
```
--horizon INT          Prediction horizon: 1, 3, or 5 days ahead (default: 3)
--results_dir PATH     Directory for results CSVs (default: results_horizon_runs)
--cache_dir PATH       Directory for cached graphs/tensors (default: cache_gcn or cache_flat_fair)
--seed INT             Random seed for reproducibility (default: 42)
```

#### `run_all_horizons_compare.py`
```
--horizons INT [...]   Space-separated horizons to run (default: 1 3 5)
--results_dir PATH     Results directory (default: results_horizon_runs)
--skip_existing        Skip if output CSV already exists
--python PATH          Python executable (default: sys.executable)
```

---

## Methodology / Approach

### Market Regime Definition

Each trading day is assigned one of three market regimes based on rolling market statistics:

| Regime | Code | Interpretation |
|--------|------|-----------------|
| **Calm** | 0 | Low volatility, high correlation (stable market) |
| **Normal** | 1 | Moderate volatility, typical correlation |
| **Turbulent** | 2 | High volatility, divergent correlations (stressed market) |

### Label Generation Pipeline

1. **Compute Rolling Features** (30-day windows):
   - **Market Volatility**: `std(daily market returns)`
   - **Average Correlation**: `mean of pairwise stock correlations`
   - **Cross-Sectional Dispersion**: `mean(std of returns across stocks)`

2. **Normalize Features** (using training set only):
   ```
   z-score normalization with train-set mean/std
   ```

3. **Create Composite Regime Score**:
   ```
   regime_score = 0.5 * volatility_z + 0.3 * correlation_z + 0.2 * dispersion_z
   ```

4. **Assign Labels** (using train-set quantiles):
   ```
   Label 0 (Calm):      regime_score <= 33rd percentile
   Label 1 (Normal):    33rd percentile < regime_score <= 66th percentile
   Label 2 (Turbulent): regime_score > 66th percentile
   ```

### Model Families

#### 1. Classical Baselines (`baselinetp_cli.py`)
Uses market-level aggregate features:
- Market return
- Market volatility
- Average correlation
- Cross-sectional dispersion

Models trained:
- **Logistic Regression**: Linear classification with regularization
- **K-Means**: Unsupervised clustering with label mapping from training set
- **Hierarchical Clustering**: Agglomerative clustering with training-set label assignment
- **Support Vector Machine (SVM)**: Non-linear classification with RBF kernel
- **Gaussian HMM**: Probabilistic sequential model for regime transitions

#### 2. Tabular Deep Learning (`flattened_node_fair_cli.py`)
No graph structure; instead, flattens per-stock features:
- Per-stock returns (100 stocks)
- Per-stock volatilities (100 stocks)
- Market-level aggregate features
- All concatenated into a single feature vector

Architecture:
- Dense layers with dropout
- LSTM for temporal sequences
- Market feature branch for joint learning
- Categorical output (3 classes)

**Purpose**: Fair neural network baseline to isolate the benefit of graph structure.

#### 3. Graph Neural Networks (`gcn_attention_cli.py`)
Explicitly models stock interdependencies:

**Graph Construction**:
- **Nodes**: 100 stocks
- **Edges**: Top-K (K=8) strongest correlations from 20-day rolling window
- **Node Features**: Per-stock returns, volatility

**Architecture**:
```
Input: Daily returns for t-window to t
  ↓
Build rolling correlation graph (top-K edges)
  ↓
GCN Layer 1 & 2 (encode node features with graph structure)
  ↓
Node Attention (importance weighting)
  ↓
Temporal Attention (across time)
  ↓
LSTM (temporal dependencies)
  ↓
Market-level features (aggregate context)
  ↓
Dense classifier (3 classes)
```

**Key Hyperparameters**:
- Window size: 20 days
- Correlation window: 20 days  
- Top-K edges: 8
- GCN layers: 2
- GCN hidden: 40
- LSTM hidden: 40
- Dropout: 0.18
- Label smoothing: 0.02
- Learning rate: 8e-4

### Training Details

**Data Splits**:
- **Train**: 2017-2022 (used for feature normalization)
- **Validation**: 2023 (hyperparameter tuning, early stopping)
- **Test**: 2024-2026 (held-out evaluation)

**Optimization**:
- Optimizer: Adam
- Loss: Cross-entropy with label smoothing
- Batch size: 128
- Epochs: 20 (with early stopping patience=5)
- Device: GPU (CUDA) if available, otherwise CPU

**Evaluation Metrics**:
- **Accuracy**: Fraction of correct predictions
- **Macro F1**: Unweighted mean of F1 scores (balances all classes)
- **Weighted F1**: Weighted by class support
- **Per-Class F1**: F1 for calm, normal, turbulent individually
- **Confusion Matrix**: Shows cross-class confusions

---

## Results / Evaluation

### Output Files Structure

After running models, check these CSV files:

1. **Per-Horizon Results**:
   - `results_horizon_runs/baselines_h{1,3,5}.csv`
   - `results_horizon_runs/flattened_fair_h{1,3,5}.csv`
   - `results_horizon_runs/gcn_attention_h{1,3,5}.csv`

2. **Consolidated Comparison**:
   - `results_horizon_runs/final_horizon_comparison.csv` — All results stacked
   - `results_horizon_runs/final_test_only_ranking.csv` — Test split ranked by Macro F1
   - `results_horizon_runs/final_test_macro_f1_pivot.csv` — Pivot: Models × Horizons

### Interpreting Results

**Expected Columns in Results CSV**:
```
horizon      — Prediction horizon (1, 3, or 5)
model        — Model name (baseline, flattened, GCN, etc.)
split        — Train/Val/Test
accuracy     — Accuracy score
macro_f1     — Unweighted F1 (primary metric)
weighted_f1  — Weighted F1
f1_calm      — F1 for calm regime
f1_normal    — F1 for normal regime
f1_turbulent — F1 for turbulent regime
```

**Key Insights**:
- **Macro F1** is the primary metric (weights all regimes equally)
- GCN models should outperform tabular baselines due to correlation structure
- Longer horizons (h=5) are harder than shorter (h=1)
- Turbulent regime is typically the most difficult to predict (fewest samples)

---

## Configuration / Parameters

### Data Configuration
**In `label_data.py`**:
```python
window = 30                    # Rolling window for features (days)
regime_score_weights = {
    'volatility': 0.5,
    'correlation': 0.3,
    'dispersion': 0.2
}
```

### Baseline Model Configuration
**In `baselinetp_cli.py`**:
```python
FEATURE_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]
```

### GCN Configuration
**In `gcn_attention_cli.py`** (Hyperparameters in `CFG` dataclass):
```python
horizon = 3                    # Prediction horizon
window = 20                    # Input sequence window
corr_window = 20              # Correlation graph window
top_k = 8                     # Top-K edges to keep
node_dim = 5                  # Node feature dimension

batch_size = 128
epochs = 20
patience = 5
lr = 8e-4
weight_decay = 1e-4
label_smoothing = 0.02

gcn_hidden = 40
gcn_layers = 2
temporal_hidden = 40
market_hidden = 24
dropout = 0.18
edge_dropout = 0.0
```

### Modifying Parameters

To experiment with different configurations:

1. **Edit source files directly** (e.g., `gcn_attention_cli.py`):
   ```python
   @dataclass
   class CFG:
       horizon: int = 5           # Change here
       lr: float = 1e-3          # And here
   ```

2. **Or pass via command line** (limited to core args):
   ```bash
   python gcn_attention_cli.py --horizon 5 --seed 123
   ```

---

## Dependencies

### Core Libraries
- **torch** (2.8.0) — Deep learning framework
- **torch.nn** — Neural network modules
- **pandas** (2.3.3) — Data manipulation
- **numpy** (2.0.2) — Numerical computing
- **scikit-learn** (1.6.1) — Classical ML models & metrics
- **scipy** (≥1.15) — Scientific computing

### Supporting Libraries
- **yfinance** (via curl_cffi, requests) — Financial data fetching
- **hmmlearn** (0.3.3) — Hidden Markov Models
- **networkx** (3.2.1) — Graph utilities
- **matplotlib** (3.9.4) — Visualization
- **tqdm** — Progress bars

### Full Requirements
See [requirements.txt](requirements.txt) for complete dependency list.

**Install all dependencies**:
```bash
pip install -r requirements.txt
```

---

## Known Issues & Troubleshooting

### Issue: GCN model fails to save outputs
**Solution**: Ensure output directories exist:
```bash
mkdir -p gcn/src/outputs gcn/src1/outputs
```

### Issue: CUDA not available (training slow)
**Solution**: The code automatically falls back to CPU. For faster training on GPU:
1. Install CUDA and cuDNN
2. Verify torch can access GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Missing data during scraping
**Solution**: Some stocks may have gaps. The `scraper.py` handles this by:
- Filling forward then backward
- Removing stocks with >50% missing data
- Documenting missingness in `data/raw_missingness.csv`

### Issue: Label distribution imbalanced
**Solution**: This is expected! Turbulent regimes are rare. The code:
- Uses macro F1 (unweighted) as the primary metric
- Applies label smoothing in neural models
- Computes confusion matrices to understand confusions

---

## Future Improvements

### Methodology Enhancements
- [ ] Multi-task learning: Predict volatility + correlation + regimes jointly
- [ ] Transformer-based temporal models (replacing LSTM)
- [ ] Attention-based edge prediction (learn graph structure end-to-end)
- [ ] Regime transition probability matrices (Markov approach)

### Data Extensions
- [ ] Add macroeconomic features (VIX, yields, credit spreads)
- [ ] Expand to international stock markets
- [ ] Include options-implied volatility data
- [ ] Sector classification for intra-market structure

### Model Variants
- [ ] Ensemble methods (stacking, boosting)
- [ ] Recurrent graph neural networks (RGCN)
- [ ] Temporal point processes for extreme events
- [ ] Uncertainty quantification (Bayesian approaches)

### Deployment
- [ ] Real-time inference pipeline
- [ ] Alert system for regime transitions
- [ ] Interactive dashboard for monitoring
- [ ] Automated retraining on new data

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and commit with clear messages
3. Ensure code passes linting (if applicable)
4. Submit a pull request with a description

### Development Setup
```bash
python -m venv venv_dev
source venv_dev/bin/activate
pip install -r requirements.txt
# Make changes...
git commit -m "Clear description of changes"
git push origin feature/your-feature-name
```

---

## License

This project is part of CSC 525 (Neural Networks and Deep Learning) coursework.  
For academic use only. Modify as needed for your research.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{market_regime_prediction_2026,
  title={Market Regime Prediction with Graph Neural Networks},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/...}}
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

## References

- **GCN Architecture**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- **Market Regimes**: Hamilton (1989) - A New Approach to the Economic Analysis of Nonstationary Time Series
- **PyTorch Documentation**: https://pytorch.org/docs/
- **scikit-learn**: https://scikit-learn.org/

---

**Last Updated**: April 2026  
**Tested On**: Python 3.8+, macOS/Linux/Windows
Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#1-1), lines 1 to 50

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#51-51), lines 51 to 100

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#101-101), lines 101 to 200

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#201-201), lines 201 to 300

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#301-301), lines 301 to 400

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#401-401), lines 401 to 500

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md#501-501), lines 501 to 600

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/requirements.txt#1-1), lines 1 to 50

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/run_all_horizons_compare.py#1-1), lines 1 to 50

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/gcn_attention_cli.py#1-1), lines 1 to 50

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/scraper.py#1-1), lines 1 to 50

Read [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/label_data.py#1-1), lines 1 to 50

Retrieving Notebook summary.

Created [](file:///Volumes/New%20Volume%20F/CSC525NNDL/MarketRegimePrediction/README.md)

# Market Regime Prediction with Graph Neural Networks

## Project Title
**Temporal Market Regime Classification: A Graph Neural Network Approach to Predicting Market Volatility Regimes**

## Overview

This project develops a machine learning pipeline for predicting **market regime transitions** (calm, normal, turbulent) using a universe of 100 US stocks. It compares classical machine learning baselines against graph neural network (GCN) models that explicitly leverage the dynamic correlation structure between stocks.

### Problem Statement
Financial markets exhibit distinct regimes characterized by different volatility and correlation dynamics. Accurate regime prediction has applications in portfolio allocation, risk management, and trading strategy design. This project tackles the problem of **classifying daily market regimes** using both tabular features and learned graph representations.

### Why This Matters
- **Risk Management**: Early detection of turbulent regimes enables portfolio hedging
- **Strategy Adaptation**: Regime-aware trading strategies can adjust parameters dynamically
- **Correlation Dynamics**: The GCN approach captures how stock correlations evolve during market stress

---

## Features

- ✅ **Automated Data Pipeline**: Yahoo Finance scraping with missingness cleaning
- ✅ **Multi-Horizon Forecasting**: Predict market regimes 1, 3, or 5 days ahead
- ✅ **Diverse Model Baselines**: Logistic Regression, K-Means, SVM, Hierarchical Clustering, Gaussian HMM
- ✅ **Graph Neural Networks**: Temporal GCN + LSTM architecture capturing correlation dynamics
- ✅ **Tabular Deep Learning**: Flattened baseline for fair comparison with neural network-based approaches
- ✅ **Comprehensive Evaluation**: Metrics across train/validation/test splits with macro-F1 focus
- ✅ **Parameterized CLI**: Easy experiment management across different horizons
- ✅ **Visualization Tools**: Regime timeline plots with market event annotations

---

## Project Structure

### Core Scripts (Root Level)

| Script | Purpose | Output |
|--------|---------|--------|
| scraper.py | Download and clean price data from Yahoo Finance | clean_prices.csv, daily_returns.csv |
| label_data.py | Compute market features and regime labels | labeled_data.csv |
| stateval.py | Validate label distributions and feature ranges | `stateval_outputs/*.csv` |
| EDAscript.py | Exploratory data analysis (optional) | `eda_outputs/*.csv` |
| turbulent_analysis.py | Analyze turbulent market periods | Console output + visualization |
| make_project_plots.py | Generate regime timeline visualizations | `plots_market_regime/*.png` |
| mapping.py | Utility for data mapping and processing | N/A |
| run_all_horizons_compare.py | Meta-script: runs all models across horizons | Consolidated comparison CSVs |

### Modeling Scripts (CLI Interface)

| Script | Models | Horizon | Output |
|--------|--------|---------|--------|
| baselinetp_cli.py | LR, K-Means, SVM, HC, HMM (classical) | Configurable | `results_horizon_runs/baselines_h{1,3,5}.csv` |
| flattened_node_fair_cli.py | Deep learning (flattened features, no graph) | Configurable | `results_horizon_runs/flattened_fair_h{1,3,5}.csv` |
| gcn_attention_cli.py | GCN + Temporal Attention + LSTM | Configurable | `results_horizon_runs/gcn_attention_h{1,3,5}.csv` |

### Directory Structure

```
MarketRegimePrediction/
├── data/                              # Raw and cleaned price/return data
│   ├── raw_prices.csv
│   ├── clean_prices.csv
│   ├── daily_returns.csv
│   └── raw_missingness.csv
├── labeled_data/                      # Market regime labels and splits
│   ├── labeled_data.csv               # Main label file (source of truth)
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── eda_outputs/                       # Exploratory analysis outputs
│   ├── return_stats.csv
│   ├── volatility_thresholds.csv
│   └── ...
├── stateval_outputs/                  # Label validation outputs
│   ├── class_distribution_by_split.csv
│   ├── train_regime_score_stats.csv
│   └── ...
├── results_horizon_runs/              # Model training results
│   ├── baselines_h1.csv, h3.csv, h5.csv
│   ├── flattened_fair_h1.csv, h3.csv, h5.csv
│   ├── gcn_attention_h1.csv, h3.csv, h5.csv
│   ├── final_horizon_comparison.csv
│   ├── final_test_only_ranking.csv
│   └── final_test_macro_f1_pivot.csv
├── plots_market_regime/               # Visualization outputs
│   └── *.png files
├── cache_gcn/                         # GCN precomputed graphs and splits
├── cache_flat_fair/                   # Flattened model cache
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── Prettyprint.ipynb                  # Notebook for result visualization
└── [core scripts mentioned above]
```

---

## Installation

### Prerequisites
- Python 3.8+
- macOS, Linux, or Windows (with PowerShell)

### Step-by-Step Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd MarketRegimePrediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verification
To verify the installation, run:
```bash
python -c "import torch; import pandas; print('Installation successful!')"
```

---

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline from raw data to model evaluation:

```bash
# 1. Data preparation
python scraper.py              # Download and clean prices
python label_data.py           # Create regime labels
python stateval.py             # Validate labels

# 2. Train all models across horizons 1, 3, 5
python run_all_horizons_compare.py

# Results are saved in results_horizon_runs/
```

### Individual Model Training

#### Train Classical Baselines
```bash
# Next-day prediction (horizon 1)
python baselinetp_cli.py --horizon 1 --results_dir results_horizon_runs

# 5-day-ahead prediction (horizon 5)
python baselinetp_cli.py --horizon 5 --results_dir results_horizon_runs
```

#### Train Flattened Tabular Deep Learning Baseline
```bash
python flattened_node_fair_cli.py --horizon 3 --results_dir results_horizon_runs
```

#### Train Graph Neural Network (GCN + Temporal Attention)
```bash
python gcn_attention_cli.py --horizon 3 --results_dir results_horizon_runs --seed 42
```

### Data Preparation Only

If you only need prepared data without training models:

```bash
python scraper.py              # Creates data/daily_returns.csv
python label_data.py           # Creates labeled_data/labeled_data.csv
python stateval.py             # Optional: validate data
```

### Exploratory Data Analysis (Optional)
```bash
python EDAscript.py                    # Generate EDA statistics
python turbulent_analysis.py           # Analyze turbulent periods
python make_project_plots.py           # Create timeline visualizations
```

### Command-Line Arguments Reference

#### baselinetp_cli.py / flattened_node_fair_cli.py / gcn_attention_cli.py
```
--horizon INT          Prediction horizon: 1, 3, or 5 days ahead (default: 3)
--results_dir PATH     Directory for results CSVs (default: results_horizon_runs)
--cache_dir PATH       Directory for cached graphs/tensors (default: cache_gcn or cache_flat_fair)
--seed INT             Random seed for reproducibility (default: 42)
```

#### run_all_horizons_compare.py
```
--horizons INT [...]   Space-separated horizons to run (default: 1 3 5)
--results_dir PATH     Results directory (default: results_horizon_runs)
--skip_existing        Skip if output CSV already exists
--python PATH          Python executable (default: sys.executable)
```

---

## Methodology / Approach

### Market Regime Definition

Each trading day is assigned one of three market regimes based on rolling market statistics:

| Regime | Code | Interpretation |
|--------|------|-----------------|
| **Calm** | 0 | Low volatility, high correlation (stable market) |
| **Normal** | 1 | Moderate volatility, typical correlation |
| **Turbulent** | 2 | High volatility, divergent correlations (stressed market) |

### Label Generation Pipeline

1. **Compute Rolling Features** (30-day windows):
   - **Market Volatility**: `std(daily market returns)`
   - **Average Correlation**: `mean of pairwise stock correlations`
   - **Cross-Sectional Dispersion**: `mean(std of returns across stocks)`

2. **Normalize Features** (using training set only):
   ```
   z-score normalization with train-set mean/std
   ```

3. **Create Composite Regime Score**:
   ```
   regime_score = 0.5 * volatility_z + 0.3 * correlation_z + 0.2 * dispersion_z
   ```

4. **Assign Labels** (using train-set quantiles):
   ```
   Label 0 (Calm):      regime_score <= 33rd percentile
   Label 1 (Normal):    33rd percentile < regime_score <= 66th percentile
   Label 2 (Turbulent): regime_score > 66th percentile
   ```

### Model Families

#### 1. Classical Baselines (baselinetp_cli.py)
Uses market-level aggregate features:
- Market return
- Market volatility
- Average correlation
- Cross-sectional dispersion

Models trained:
- **Logistic Regression**: Linear classification with regularization
- **K-Means**: Unsupervised clustering with label mapping from training set
- **Hierarchical Clustering**: Agglomerative clustering with training-set label assignment
- **Support Vector Machine (SVM)**: Non-linear classification with RBF kernel
- **Gaussian HMM**: Probabilistic sequential model for regime transitions

#### 2. Tabular Deep Learning (flattened_node_fair_cli.py)
No graph structure; instead, flattens per-stock features:
- Per-stock returns (100 stocks)
- Per-stock volatilities (100 stocks)
- Market-level aggregate features
- All concatenated into a single feature vector

Architecture:
- Dense layers with dropout
- LSTM for temporal sequences
- Market feature branch for joint learning
- Categorical output (3 classes)

**Purpose**: Fair neural network baseline to isolate the benefit of graph structure.

#### 3. Graph Neural Networks (gcn_attention_cli.py)
Explicitly models stock interdependencies:

**Graph Construction**:
- **Nodes**: 100 stocks
- **Edges**: Top-K (K=8) strongest correlations from 20-day rolling window
- **Node Features**: Per-stock returns, volatility

**Architecture**:
```
Input: Daily returns for t-window to t
  ↓
Build rolling correlation graph (top-K edges)
  ↓
GCN Layer 1 & 2 (encode node features with graph structure)
  ↓
Node Attention (importance weighting)
  ↓
Temporal Attention (across time)
  ↓
LSTM (temporal dependencies)
  ↓
Market-level features (aggregate context)
  ↓
Dense classifier (3 classes)
```

**Key Hyperparameters**:
- Window size: 20 days
- Correlation window: 20 days  
- Top-K edges: 8
- GCN layers: 2
- GCN hidden: 40
- LSTM hidden: 40
- Dropout: 0.18
- Label smoothing: 0.02
- Learning rate: 8e-4

### Training Details

**Data Splits**:
- **Train**: 2017-2022 (used for feature normalization)
- **Validation**: 2023 (hyperparameter tuning, early stopping)
- **Test**: 2024-2026 (held-out evaluation)

**Optimization**:
- Optimizer: Adam
- Loss: Cross-entropy with label smoothing
- Batch size: 128
- Epochs: 20 (with early stopping patience=5)
- Device: GPU (CUDA) if available, otherwise CPU

**Evaluation Metrics**:
- **Accuracy**: Fraction of correct predictions
- **Macro F1**: Unweighted mean of F1 scores (balances all classes)
- **Weighted F1**: Weighted by class support
- **Per-Class F1**: F1 for calm, normal, turbulent individually
- **Confusion Matrix**: Shows cross-class confusions

---

## Results / Evaluation

### Output Files Structure

After running models, check these CSV files:

1. **Per-Horizon Results**:
   - `results_horizon_runs/baselines_h{1,3,5}.csv`
   - `results_horizon_runs/flattened_fair_h{1,3,5}.csv`
   - `results_horizon_runs/gcn_attention_h{1,3,5}.csv`

2. **Consolidated Comparison**:
   - final_horizon_comparison.csv — All results stacked
   - final_test_only_ranking.csv — Test split ranked by Macro F1
   - final_test_macro_f1_pivot.csv — Pivot: Models × Horizons

### Interpreting Results

**Expected Columns in Results CSV**:
```
horizon      — Prediction horizon (1, 3, or 5)
model        — Model name (baseline, flattened, GCN, etc.)
split        — Train/Val/Test
accuracy     — Accuracy score
macro_f1     — Unweighted F1 (primary metric)
weighted_f1  — Weighted F1
f1_calm      — F1 for calm regime
f1_normal    — F1 for normal regime
f1_turbulent — F1 for turbulent regime
```

**Key Insights**:
- **Macro F1** is the primary metric (weights all regimes equally)
- GCN models should outperform tabular baselines due to correlation structure
- Longer horizons (h=5) are harder than shorter (h=1)
- Turbulent regime is typically the most difficult to predict (fewest samples)

---

## Configuration / Parameters

### Data Configuration
**In label_data.py**:
```python
window = 30                    # Rolling window for features (days)
regime_score_weights = {
    'volatility': 0.5,
    'correlation': 0.3,
    'dispersion': 0.2
}
```

### Baseline Model Configuration
**In baselinetp_cli.py**:
```python
FEATURE_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]
```

### GCN Configuration
**In gcn_attention_cli.py** (Hyperparameters in `CFG` dataclass):
```python
horizon = 3                    # Prediction horizon
window = 20                    # Input sequence window
corr_window = 20              # Correlation graph window
top_k = 8                     # Top-K edges to keep
node_dim = 5                  # Node feature dimension

batch_size = 128
epochs = 20
patience = 5
lr = 8e-4
weight_decay = 1e-4
label_smoothing = 0.02

gcn_hidden = 40
gcn_layers = 2
temporal_hidden = 40
market_hidden = 24
dropout = 0.18
edge_dropout = 0.0
```

### Modifying Parameters

To experiment with different configurations:

1. **Edit source files directly** (e.g., gcn_attention_cli.py):
   ```python
   @dataclass
   class CFG:
       horizon: int = 5           # Change here
       lr: float = 1e-3          # And here
   ```

2. **Or pass via command line** (limited to core args):
   ```bash
   python gcn_attention_cli.py --horizon 5 --seed 123
   ```

---

## Dependencies

### Core Libraries
- **torch** (2.8.0) — Deep learning framework
- **torch.nn** — Neural network modules
- **pandas** (2.3.3) — Data manipulation
- **numpy** (2.0.2) — Numerical computing
- **scikit-learn** (1.6.1) — Classical ML models & metrics
- **scipy** (≥1.15) — Scientific computing

### Supporting Libraries
- **yfinance** (via curl_cffi, requests) — Financial data fetching
- **hmmlearn** (0.3.3) — Hidden Markov Models
- **networkx** (3.2.1) — Graph utilities
- **matplotlib** (3.9.4) — Visualization
- **tqdm** — Progress bars

### Full Requirements
See requirements.txt for complete dependency list.

**Install all dependencies**:
```bash
pip install -r requirements.txt
```

---

## Known Issues & Troubleshooting

### Issue: GCN model fails to save outputs
**Solution**: Ensure output directories exist:
```bash
mkdir -p gcn/src/outputs gcn/src1/outputs
```

### Issue: CUDA not available (training slow)
**Solution**: The code automatically falls back to CPU. For faster training on GPU:
1. Install CUDA and cuDNN
2. Verify torch can access GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Missing data during scraping
**Solution**: Some stocks may have gaps. The scraper.py handles this by:
- Filling forward then backward
- Removing stocks with >50% missing data
- Documenting missingness in raw_missingness.csv

### Issue: Label distribution imbalanced
**Solution**: This is expected! Turbulent regimes are rare. The code:
- Uses macro F1 (unweighted) as the primary metric
- Applies label smoothing in neural models
- Computes confusion matrices to understand confusions

---

## Future Improvements

### Methodology Enhancements
- [ ] Multi-task learning: Predict volatility + correlation + regimes jointly
- [ ] Transformer-based temporal models (replacing LSTM)
- [ ] Attention-based edge prediction (learn graph structure end-to-end)
- [ ] Regime transition probability matrices (Markov approach)

### Data Extensions
- [ ] Add macroeconomic features (VIX, yields, credit spreads)
- [ ] Expand to international stock markets
- [ ] Include options-implied volatility data
- [ ] Sector classification for intra-market structure

### Model Variants
- [ ] Ensemble methods (stacking, boosting)
- [ ] Recurrent graph neural networks (RGCN)
- [ ] Temporal point processes for extreme events
- [ ] Uncertainty quantification (Bayesian approaches)

### Deployment
- [ ] Real-time inference pipeline
- [ ] Alert system for regime transitions
- [ ] Interactive dashboard for monitoring
- [ ] Automated retraining on new data

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and commit with clear messages
3. Ensure code passes linting (if applicable)
4. Submit a pull request with a description

### Development Setup
```bash
python -m venv venv_dev
source venv_dev/bin/activate
pip install -r requirements.txt
# Make changes...
git commit -m "Clear description of changes"
git push origin feature/your-feature-name
```

---

## License

This project is part of CSC 525 (Neural Networks and Deep Learning) coursework.  
For academic use only. Modify as needed for your research.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{market_regime_prediction_2026,
  title={Market Regime Prediction with Graph Neural Networks},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/...}}
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

## References

- **GCN Architecture**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- **Market Regimes**: Hamilton (1989) - A New Approach to the Economic Analysis of Nonstationary Time Series
- **PyTorch Documentation**: https://pytorch.org/docs/
- **scikit-learn**: https://scikit-learn.org/

---You've used 56% of your session rate limit. Your session rate limit will reset on April 29 at 7:31 AM. [Learn More](https://aka.ms/github-copilot-rate-limit-error)