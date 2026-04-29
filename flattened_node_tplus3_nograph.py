"""
flattened_node_tplus3_nograph.py

No-graph flattened-node baseline for market regime prediction.

Purpose:
    This is the ablation baseline for train_gat_attention_tplus3.py.
    It uses the same t+3 target and chronological split, and it uses stock-level
    rolling features, but it deliberately removes the graph/correlation topology.

Interpretation:
    If the GAT/graph-attention model beats this model, that supports the claim
    that explicit graph structure is adding useful information beyond flattened
    stock features.

Run from project root:
    python flattened_node_tplus3_nograph.py

Expected files:
    data/daily_returns.csv
    labeled_data/labeled_data.csv
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


@dataclass
class CFG:
    horizon: int = 3
    window: int = 30
    batch_size: int = 32
    epochs: int = 80
    patience: int = 14
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden: int = 160
    temporal_hidden: int = 96
    dropout: float = 0.35
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(cfg.seed)


MARKET_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]


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
    returns = returns.loc[labels.index].copy()
    return returns, labels


def node_features_for_day(returns: pd.DataFrame, pos: int) -> np.ndarray:
    """
    Per-stock features, no graph edges:
      - 1-day return
      - 5-day cumulative return
      - 10-day cumulative return
      - 30-day volatility
      - excess return over market return
    Output shape: [num_stocks, 5]
    """
    r1 = returns.iloc[pos].values
    r5 = returns.iloc[pos - 4 : pos + 1].sum(axis=0).values
    r10 = returns.iloc[pos - 9 : pos + 1].sum(axis=0).values
    vol30 = returns.iloc[pos - 29 : pos + 1].std(axis=0).values
    excess = r1 - np.nanmean(r1)
    x = np.stack([r1, r5, r10, vol30, excess], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def valid_dates(labels: pd.DataFrame) -> List[pd.Timestamp]:
    return list(labels.index[cfg.window:])


class FlattenedNodeDataset(Dataset):
    def __init__(
        self,
        returns: pd.DataFrame,
        labels: pd.DataFrame,
        dates: List[pd.Timestamp],
        node_scaler: StandardScaler,
        market_scaler: StandardScaler,
    ):
        self.returns = returns
        self.labels = labels
        self.dates = dates
        self.node_scaler = node_scaler
        self.market_scaler = market_scaler
        self.date_to_pos = {d: i for i, d in enumerate(labels.index)}

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int):
        d = self.dates[idx]
        pos = self.date_to_pos[d]

        # Sequence of flattened stock features: [window, num_stocks * 5]
        seq = []
        for t in range(pos - cfg.window + 1, pos + 1):
            x = node_features_for_day(self.returns, t)       # [N, 5]
            x = self.node_scaler.transform(x)                # scale feature columns across stock-days
            seq.append(x.reshape(-1))                        # flatten, removes topology
        seq = np.stack(seq, axis=0).astype(np.float32)

        market = self.labels.loc[d, MARKET_COLS].values.reshape(1, -1)
        market = self.market_scaler.transform(market).astype(np.float32).squeeze(0)
        y = int(self.labels.loc[d, "future_label"])

        return torch.tensor(seq), torch.tensor(market), torch.tensor(y, dtype=torch.long)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D]
        weights = torch.softmax(self.score(h), dim=1)         # [B, T, 1]
        return (h * weights).sum(dim=1)


class FlattenedNodeTemporalModel(nn.Module):
    def __init__(self, flat_dim: int, market_dim: int = 4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(flat_dim, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.temporal_hidden),
            nn.LayerNorm(cfg.temporal_hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=cfg.temporal_hidden,
            hidden_size=cfg.temporal_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attn = TemporalAttention(cfg.temporal_hidden * 2)
        self.market_net = nn.Sequential(
            nn.Linear(market_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.temporal_hidden * 2 + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 3),
        )

    def forward(self, seq: torch.Tensor, market: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, flat_dim]
        z = self.input_proj(seq)
        h, _ = self.gru(z)
        h = self.temporal_attn(h)
        m = self.market_net(market)
        return self.classifier(torch.cat([h, m], dim=1))


def split_dates(dates: List[pd.Timestamp]):
    train = [d for d in dates if d.year <= 2022]
    val = [d for d in dates if d.year == 2023]
    test = [d for d in dates if d.year >= 2024]
    return train, val, test


def fit_scalers(returns: pd.DataFrame, labels: pd.DataFrame, train_dates: List[pd.Timestamp]):
    date_to_pos = {d: i for i, d in enumerate(labels.index)}

    node_rows = []
    for d in train_dates:
        pos = date_to_pos[d]
        for t in range(pos - cfg.window + 1, pos + 1):
            node_rows.append(node_features_for_day(returns, t))
    node_rows = np.concatenate(node_rows, axis=0)

    node_scaler = StandardScaler()
    node_scaler.fit(node_rows)

    market_scaler = StandardScaler()
    market_scaler.fit(labels.loc[train_dates, MARKET_COLS].values)
    return node_scaler, market_scaler


def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)


def class_weights(labels: pd.DataFrame, train_dates: List[pd.Timestamp]) -> torch.Tensor:
    y = labels.loc[train_dates, "future_label"].values
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    print("Class counts:", counts.astype(int).tolist())
    print("Class weights:", np.round(weights, 4).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=cfg.device)


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for seq, market, y in loader:
            seq = seq.to(cfg.device)
            market = market.to(cfg.device)
            y = y.to(cfg.device)
            logits = model(seq, market)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_report(name: str, result: Dict):
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Accuracy    : {result['acc']:.4f}")
    print(f"Macro F1    : {result['macro_f1']:.4f}")
    print(f"Weighted F1 : {result['weighted_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(result["y_true"], result["y_pred"], labels=[0, 1, 2]))
    print("\nClassification Report:")
    print(classification_report(result["y_true"], result["y_pred"], labels=[0, 1, 2], zero_division=0))


def main() -> None:
    print("Device:", cfg.device)
    returns, labels = load_data()
    dates = valid_dates(labels)
    train_dates, val_dates, test_dates = split_dates(dates)

    print("Returns shape:", returns.shape)
    print("Labels shape :", labels.shape)
    print("Samples: train", len(train_dates), "val", len(val_dates), "test", len(test_dates))
    print("Horizon:", cfg.horizon)

    node_scaler, market_scaler = fit_scalers(returns, labels, train_dates)

    train_ds = FlattenedNodeDataset(returns, labels, train_dates, node_scaler, market_scaler)
    val_ds = FlattenedNodeDataset(returns, labels, val_dates, node_scaler, market_scaler)
    test_ds = FlattenedNodeDataset(returns, labels, test_dates, node_scaler, market_scaler)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    sample_seq, sample_market, _ = train_ds[0]
    model = FlattenedNodeTemporalModel(flat_dim=sample_seq.shape[-1], market_dim=sample_market.shape[-1]).to(cfg.device)

    weights = class_weights(labels, train_dates)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for seq, market, y in train_loader:
            seq = seq.to(cfg.device)
            market = market.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(seq, market)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        val_result = evaluate(model, val_loader)
        scheduler.step(val_result["macro_f1"])
        train_loss /= len(train_loader.dataset)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Macro F1 {val_result['macro_f1']:.4f} | "
            f"Val Acc {val_result['acc']:.4f}"
        )

        if val_result["macro_f1"] > best_val:
            best_val = val_result["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_result = evaluate(model, val_loader)
    test_result = evaluate(model, test_loader)
    print_report("Flattened Node No-Graph Temporal Attention - Validation", val_result)
    print_report("Flattened Node No-Graph Temporal Attention - Test", test_result)

    print("\nAblation interpretation:")
    print("This model sees stock-level features but no explicit correlation graph edges.")
    print("If train_gat_attention_tplus3.py beats this, use the gap as evidence that graph structure helps.")


if __name__ == "__main__":
    main()
