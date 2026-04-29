"""
train_gat_attention_tplus3.py

Hybrid Graph Attention + Temporal Attention model for market regime prediction.
Goal: predict future regime label at t+3 using:
  1) stock-level graph sequence from daily returns
  2) market-level features used by the strong LR/SVM baselines

Run from the MarketRegimePrediction project root:
    python train_gat_attention_tplus3.py

Expected files:
    data/daily_returns.csv
    labeled_data/labeled_data.csv

This script intentionally uses the same chronological split as your baseline:
    Train: 2017-2022
    Val:   2023
    Test:  2024-2026
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
except Exception as exc:
    raise ImportError(
        "torch-geometric is required. Install it first, then rerun. "
        "Your requirements.txt already lists torch-geometric, so usually `pip install -r requirements.txt` is enough."
    ) from exc


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    horizon: int = 3
    window: int = 30              # graph sequence length
    corr_window: int = 30         # rolling correlation window for edges
    top_k: int = 8                # top correlated neighbors per stock
    batch_size: int = 16
    epochs: int = 80
    patience: int = 14
    lr: float = 8e-4
    weight_decay: float = 1e-4
    gat_hidden: int = 48
    gat_out: int = 48
    gat_heads: int = 3
    temporal_hidden: int = 96
    market_hidden: int = 48
    dropout: float = 0.25
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(cfg.seed)


# -----------------------------
# Metrics
# -----------------------------
def print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro:.4f}")
    print(f"Weighted F1 : {weighted:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0))
    return {"acc": acc, "macro_f1": macro, "weighted_f1": weighted}


# -----------------------------
# Data loading and feature setup
# -----------------------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)
    labels = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

    common_idx = returns.index.intersection(labels.index)
    returns = returns.loc[common_idx].copy().sort_index()
    labels = labels.loc[common_idx].copy().sort_index()

    required = ["market_volatility", "avg_correlation", "cross_sectional_dispersion", "label"]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Missing required columns in labeled_data.csv: {missing}")

    labels["market_return"] = returns.mean(axis=1)
    labels["future_label"] = labels["label"].shift(-cfg.horizon)
    labels = labels.dropna(subset=["future_label"]).copy()
    labels["future_label"] = labels["future_label"].astype(int)

    # keep aligned after horizon drop
    returns = returns.loc[labels.index].copy()
    return returns, labels


MARKET_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]


def make_sample_indices(returns: pd.DataFrame, labels: pd.DataFrame) -> List[pd.Timestamp]:
    # Need enough history for node features and correlation graph.
    min_i = max(cfg.window, cfg.corr_window)
    valid_dates = []
    for i in range(min_i, len(labels)):
        d = labels.index[i]
        # future label already exists due to earlier dropna
        valid_dates.append(d)
    return valid_dates


# -----------------------------
# Graph construction
# -----------------------------
def corr_to_edge_index_and_attr(corr: np.ndarray, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds directed top-k graph from absolute correlation.
    Edge attribute stores signed correlation.
    """
    n = corr.shape[0]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    src, dst, attrs = [], [], []
    abs_corr = np.abs(corr)
    for i in range(n):
        neigh = np.argsort(abs_corr[i])[-top_k:]
        for j in neigh:
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            attrs.append(corr[i, j])

    # Add self loops manually with attr 1.0. GATv2Conv can also add self loops,
    # but explicit loops keep edge_attr dimensionality clean.
    for i in range(n):
        src.append(i)
        dst.append(i)
        attrs.append(1.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_attr


class MarketGraphDataset(Dataset):
    def __init__(
        self,
        returns: pd.DataFrame,
        labels: pd.DataFrame,
        dates: List[pd.Timestamp],
        market_scaler: StandardScaler,
        node_scaler: StandardScaler,
    ):
        self.returns = returns
        self.labels = labels
        self.dates = dates
        self.market_scaler = market_scaler
        self.node_scaler = node_scaler
        self.date_to_pos = {d: i for i, d in enumerate(labels.index)}
        self.tickers = list(returns.columns)
        self.n_nodes = len(self.tickers)

    def __len__(self) -> int:
        return len(self.dates)

    def _node_features_for_day(self, pos: int) -> np.ndarray:
        """
        Node features for each stock at one day t:
          - r_1d
          - rolling 5d sum return
          - rolling 10d sum return
          - rolling 30d volatility
          - stock return minus market return
        """
        r = self.returns
        ret_1d = r.iloc[pos].values
        ret_5d = r.iloc[pos - 4: pos + 1].sum(axis=0).values
        ret_10d = r.iloc[pos - 9: pos + 1].sum(axis=0).values
        vol_30d = r.iloc[pos - 29: pos + 1].std(axis=0).values
        excess = ret_1d - r.iloc[pos].mean()
        x = np.stack([ret_1d, ret_5d, ret_10d, vol_30d, excess], axis=1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.node_scaler.transform(x)
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        date = self.dates[idx]
        pos = self.date_to_pos[date]

        xs, edge_indices, edge_attrs = [], [], []
        for t in range(pos - cfg.window + 1, pos + 1):
            x_t = self._node_features_for_day(t)
            corr_start = t - cfg.corr_window + 1
            corr = self.returns.iloc[corr_start: t + 1].corr().values
            edge_index, edge_attr = corr_to_edge_index_and_attr(corr, cfg.top_k)
            xs.append(torch.tensor(x_t, dtype=torch.float32))
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        market = self.labels.loc[date, MARKET_COLS].values.astype(np.float32).reshape(1, -1)
        market = self.market_scaler.transform(market).astype(np.float32).squeeze(0)
        y = int(self.labels.loc[date, "future_label"])

        return {
            "x_seq": xs,                         # list length window, each [N, F]
            "edge_index_seq": edge_indices,       # list length window
            "edge_attr_seq": edge_attrs,          # list length window
            "market": torch.tensor(market, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }


def collate_graph_sequences(batch: List[Dict]):
    # Keep list structure because each sample has a sequence of graph objects.
    return batch


# -----------------------------
# Model
# -----------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=cfg.gat_hidden,
            heads=cfg.gat_heads,
            concat=True,
            edge_dim=1,
            dropout=cfg.dropout,
            add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            in_channels=cfg.gat_hidden * cfg.gat_heads,
            out_channels=cfg.gat_out,
            heads=1,
            concat=False,
            edge_dim=1,
            dropout=cfg.dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(cfg.gat_hidden * cfg.gat_heads)
        self.norm2 = nn.LayerNorm(cfg.gat_out)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = F.elu(h)

        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return pooled  # [1, 2 * gat_out]


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, h_seq: torch.Tensor):
        # h_seq: [B, T, D]
        scores = self.score(h_seq).squeeze(-1)       # [B, T]
        weights = torch.softmax(scores, dim=1)       # [B, T]
        context = torch.sum(h_seq * weights.unsqueeze(-1), dim=1)
        return context, weights


class GATTemporalMarketFusion(nn.Module):
    def __init__(self, node_in_dim: int, market_in_dim: int, num_classes: int = 3):
        super().__init__()
        self.graph_encoder = GraphEncoder(node_in_dim)
        graph_dim = 2 * cfg.gat_out

        self.temporal = nn.GRU(
            input_size=graph_dim,
            hidden_size=cfg.temporal_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.temporal_attn = TemporalAttention(2 * cfg.temporal_hidden)

        self.market_net = nn.Sequential(
            nn.Linear(market_in_dim, cfg.market_hidden),
            nn.LayerNorm(cfg.market_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.market_hidden, cfg.market_hidden),
            nn.ReLU(),
        )

        fused_dim = (2 * cfg.temporal_hidden) + cfg.market_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch: List[Dict]):
        device = next(self.parameters()).device
        seq_embeddings = []
        market_features = []

        for sample in batch:
            day_embeddings = []
            for x, edge_index, edge_attr in zip(
                sample["x_seq"], sample["edge_index_seq"], sample["edge_attr_seq"]
            ):
                x = x.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                emb = self.graph_encoder(x, edge_index, edge_attr)  # [1, graph_dim]
                day_embeddings.append(emb.squeeze(0))
            seq_embeddings.append(torch.stack(day_embeddings, dim=0))
            market_features.append(sample["market"].to(device))

        h_seq = torch.stack(seq_embeddings, dim=0)      # [B, T, graph_dim]
        market = torch.stack(market_features, dim=0)    # [B, market_dim]

        temporal_out, _ = self.temporal(h_seq)          # [B, T, 2H]
        temporal_ctx, attn_weights = self.temporal_attn(temporal_out)
        market_ctx = self.market_net(market)

        fused = torch.cat([temporal_ctx, market_ctx], dim=1)
        logits = self.classifier(fused)
        return logits, attn_weights


# -----------------------------
# Train/eval helpers
# -----------------------------
def get_targets(batch: List[Dict], device: str):
    return torch.stack([s["y"] for s in batch]).to(device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        y = get_targets(batch, device)
        logits, _ = model(batch)
        pred = torch.argmax(logits, dim=1)
        ys.extend(y.cpu().numpy())
        preds.extend(pred.cpu().numpy())
    return np.array(ys), np.array(preds)


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        y = get_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * len(batch)
        n += len(batch)
    return total_loss / max(n, 1)


def class_weights_from_y(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    # Smooth weights so class 2 does not dominate too aggressively.
    weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    print("Device:", cfg.device)
    print("Config:", cfg)

    returns, labels = load_data()
    all_dates = make_sample_indices(returns, labels)

    train_dates = [d for d in all_dates if d.year <= 2022]
    val_dates = [d for d in all_dates if d.year == 2023]
    test_dates = [d for d in all_dates if d.year >= 2024]

    print("\nSample counts:")
    print("Train:", len(train_dates))
    print("Val:  ", len(val_dates))
    print("Test: ", len(test_dates))

    # Fit scalers on training only.
    market_scaler = StandardScaler()
    market_scaler.fit(labels.loc[train_dates, MARKET_COLS].values)

    # Node scaler: collect node features from train dates only.
    tmp_date_to_pos = {d: i for i, d in enumerate(labels.index)}
    node_rows = []
    for d in train_dates:
        pos = tmp_date_to_pos[d]
        ret_1d = returns.iloc[pos].values
        ret_5d = returns.iloc[pos - 4: pos + 1].sum(axis=0).values
        ret_10d = returns.iloc[pos - 9: pos + 1].sum(axis=0).values
        vol_30d = returns.iloc[pos - 29: pos + 1].std(axis=0).values
        excess = ret_1d - returns.iloc[pos].mean()
        x = np.stack([ret_1d, ret_5d, ret_10d, vol_30d, excess], axis=1)
        node_rows.append(x)
    node_rows = np.concatenate(node_rows, axis=0)
    node_rows = np.nan_to_num(node_rows, nan=0.0, posinf=0.0, neginf=0.0)
    node_scaler = StandardScaler()
    node_scaler.fit(node_rows)

    train_ds = MarketGraphDataset(returns, labels, train_dates, market_scaler, node_scaler)
    val_ds = MarketGraphDataset(returns, labels, val_dates, market_scaler, node_scaler)
    test_ds = MarketGraphDataset(returns, labels, test_dates, market_scaler, node_scaler)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_graph_sequences
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_graph_sequences
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_graph_sequences
    )

    y_train = labels.loc[train_dates, "future_label"].values
    print("\nTrain class counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val class counts:  ", dict(zip(*np.unique(labels.loc[val_dates, "future_label"].values, return_counts=True))))
    print("Test class counts: ", dict(zip(*np.unique(labels.loc[test_dates, "future_label"].values, return_counts=True))))

    weights = class_weights_from_y(y_train).to(cfg.device)
    print("Class weights:", weights.detach().cpu().numpy())

    model = GATTemporalMarketFusion(node_in_dim=5, market_in_dim=len(MARKET_COLS)).to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        yv, pv = predict(model, val_loader, cfg.device)
        val_macro = f1_score(yv, pv, average="macro", zero_division=0)
        val_acc = accuracy_score(yv, pv)
        scheduler.step(val_macro)

        print(f"Epoch {epoch:03d} | loss {loss:.4f} | val_acc {val_acc:.4f} | val_macro_f1 {val_macro:.4f}")

        if val_macro > best_val + 1e-5:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch}. Best val macro F1: {best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    yv, pv = predict(model, val_loader, cfg.device)
    yt, pt = predict(model, test_loader, cfg.device)

    print_eval("GAT + Temporal Attention + Market Fusion (t+3) - Validation", yv, pv)
    print_eval("GAT + Temporal Attention + Market Fusion (t+3) - Test", yt, pt)

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "market_cols": MARKET_COLS,
            "tickers": list(returns.columns),
            "best_val_macro_f1": best_val,
        },
        out_dir / "best_gat_attention_tplus3.pt",
    )
    print("\nSaved checkpoint: checkpoints/best_gat_attention_tplus3.pt")


if __name__ == "__main__":
    main()
