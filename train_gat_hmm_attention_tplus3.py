"""
train_gat_hmm_attention_tplus3.py

Upgraded hybrid model for Learning Market Regimes from Time-Evolving Asset Graphs.

Predicts future regime label at t+3 using:
  1) time-evolving stock correlation graphs
  2) GATv2 graph convolution/attention
  3) GRU + temporal attention over the graph sequence
  4) market-feature fusion
  5) HMM latent-regime features fit only on the training period

Run from project root:
    python train_gat_hmm_attention_tplus3.py
"""

from __future__ import annotations

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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
except Exception as exc:
    raise ImportError("torch-geometric is required. Run: pip install -r requirements.txt") from exc

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None
    HMM_AVAILABLE = False


@dataclass
class CFG:
    horizon: int = 3
    window: int = 30
    corr_window: int = 30
    top_k: int = 8
    batch_size: int = 12
    epochs: int = 100
    patience: int = 18
    lr: float = 6e-4
    weight_decay: float = 8e-5
    gat_hidden: int = 64
    gat_out: int = 64
    gat_heads: int = 4
    temporal_hidden: int = 128
    market_hidden: int = 64
    hmm_components: int = 5
    hmm_iter: int = 200
    focal_gamma: float = 1.5
    dropout: float = 0.22
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()

MARKET_COLS = [
    "market_return",
    "market_volatility",
    "avg_correlation",
    "cross_sectional_dispersion",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)
    labels = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)
    common_idx = returns.index.intersection(labels.index)
    returns = returns.loc[common_idx].sort_index().copy()
    labels = labels.loc[common_idx].sort_index().copy()

    required = ["market_volatility", "avg_correlation", "cross_sectional_dispersion", "label"]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Missing columns in labeled_data.csv: {missing}")

    labels["market_return"] = returns.mean(axis=1)
    labels["future_label"] = labels["label"].shift(-cfg.horizon)
    labels = labels.dropna(subset=["future_label"]).copy()
    labels["future_label"] = labels["future_label"].astype(int)
    returns = returns.loc[labels.index].copy()
    return returns, labels


def make_sample_indices(labels: pd.DataFrame) -> List[pd.Timestamp]:
    min_i = max(cfg.window, cfg.corr_window)
    return list(labels.index[min_i:])


def corr_to_edge_index_and_attr(corr: np.ndarray, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = corr.shape[0]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)

    src, dst, attrs = [], [], []
    for i in range(n):
        neigh = np.argsort(abs_corr[i])[-top_k:]
        for j in neigh:
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            attrs.append(corr[i, j])

    for i in range(n):
        src.append(i)
        dst.append(i)
        attrs.append(1.0)

    return (
        torch.tensor([src, dst], dtype=torch.long),
        torch.tensor(attrs, dtype=torch.float32).view(-1, 1),
    )


class MarketGraphDataset(Dataset):
    def __init__(
        self,
        returns: pd.DataFrame,
        labels: pd.DataFrame,
        dates: List[pd.Timestamp],
        market_scaler: StandardScaler,
        node_scaler: StandardScaler,
        hmm_features: pd.DataFrame | None = None,
    ):
        self.returns = returns
        self.labels = labels
        self.dates = dates
        self.market_scaler = market_scaler
        self.node_scaler = node_scaler
        self.hmm_features = hmm_features
        self.date_to_pos = {d: i for i, d in enumerate(labels.index)}
        self.n_nodes = returns.shape[1]

    def __len__(self) -> int:
        return len(self.dates)

    def _node_features_for_day(self, pos: int) -> np.ndarray:
        r = self.returns
        ret_1d = r.iloc[pos].values
        ret_5d = r.iloc[pos - 4 : pos + 1].sum(axis=0).values
        ret_10d = r.iloc[pos - 9 : pos + 1].sum(axis=0).values
        vol_30d = r.iloc[pos - 29 : pos + 1].std(axis=0).values
        excess = ret_1d - r.iloc[pos].mean()
        x = np.stack([ret_1d, ret_5d, ret_10d, vol_30d, excess], axis=1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.node_scaler.transform(x).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        date = self.dates[idx]
        pos = self.date_to_pos[date]

        xs, edge_indices, edge_attrs = [], [], []
        for t in range(pos - cfg.window + 1, pos + 1):
            x_t = self._node_features_for_day(t)
            corr = self.returns.iloc[t - cfg.corr_window + 1 : t + 1].corr().values
            edge_index, edge_attr = corr_to_edge_index_and_attr(corr, cfg.top_k)
            xs.append(torch.tensor(x_t, dtype=torch.float32))
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        market = self.labels.loc[date, MARKET_COLS].values.astype(np.float32).reshape(1, -1)
        market = self.market_scaler.transform(market).astype(np.float32).squeeze(0)

        if self.hmm_features is not None and self.hmm_features.shape[1] > 0:
            hmm = self.hmm_features.loc[date].values.astype(np.float32)
        else:
            hmm = np.zeros(0, dtype=np.float32)

        return {
            "x_seq": xs,
            "edge_index_seq": edge_indices,
            "edge_attr_seq": edge_attrs,
            "market": torch.tensor(market, dtype=torch.float32),
            "hmm": torch.tensor(hmm, dtype=torch.float32),
            "y": torch.tensor(int(self.labels.loc[date, "future_label"]), dtype=torch.long),
        }


def collate_graph_sequences(batch: List[Dict]) -> List[Dict]:
    return batch


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, gamma: float = 1.5, label_smoothing: float = 0.02):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


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
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_attr)
        h = self.drop(F.elu(self.norm1(h)))
        h = self.gat2(h, edge_index, edge_attr)
        h = F.elu(self.norm2(h))
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(h_seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(h_seq * weights.unsqueeze(-1), dim=1), weights


class GATTemporalHMMMarketFusion(nn.Module):
    def __init__(self, node_in_dim: int, market_in_dim: int, hmm_in_dim: int, num_classes: int = 3):
        super().__init__()
        self.graph_encoder = GraphEncoder(node_in_dim)
        graph_dim = 2 * cfg.gat_out

        self.temporal = nn.GRU(
            input_size=graph_dim,
            hidden_size=cfg.temporal_hidden,
            batch_first=True,
            bidirectional=True,
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

        self.hmm_hidden = 32 if hmm_in_dim > 0 else 0
        self.hmm_net = None
        if hmm_in_dim > 0:
            self.hmm_net = nn.Sequential(
                nn.Linear(hmm_in_dim, self.hmm_hidden),
                nn.LayerNorm(self.hmm_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )

        fused_dim = 2 * cfg.temporal_hidden + cfg.market_hidden + self.hmm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(80, num_classes),
        )

    def forward(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        seq_embeddings, market_features, hmm_features = [], [], []

        for sample in batch:
            day_embeddings = []
            for x, edge_index, edge_attr in zip(sample["x_seq"], sample["edge_index_seq"], sample["edge_attr_seq"]):
                emb = self.graph_encoder(x.to(device), edge_index.to(device), edge_attr.to(device))
                day_embeddings.append(emb.squeeze(0))
            seq_embeddings.append(torch.stack(day_embeddings, dim=0))
            market_features.append(sample["market"].to(device))
            if self.hmm_net is not None:
                hmm_features.append(sample["hmm"].to(device))

        h_seq = torch.stack(seq_embeddings, dim=0)
        temporal_out, _ = self.temporal(h_seq)
        temporal_ctx, attn_weights = self.temporal_attn(temporal_out)

        market = torch.stack(market_features, dim=0)
        parts = [temporal_ctx, self.market_net(market)]
        if self.hmm_net is not None:
            hmm = torch.stack(hmm_features, dim=0)
            parts.append(self.hmm_net(hmm))

        logits = self.classifier(torch.cat(parts, dim=1))
        return logits, attn_weights


def get_targets(batch: List[Dict], device: str) -> torch.Tensor:
    return torch.stack([s["y"] for s in batch]).to(device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        y = get_targets(batch, device)
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy())
        preds.extend(pred.cpu().numpy())
    return np.array(ys), np.array(preds)


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        y = get_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total += loss.item() * len(batch)
        n += len(batch)
    return total / max(n, 1)


def print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro:.4f}")
    print(f"Weighted F1 : {weighted:.4f}")
    print(f"Per-class F1: calm={per[0]:.4f}, normal={per[1]:.4f}, turbulent={per[2]:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0))
    return {"acc": acc, "macro_f1": macro, "weighted_f1": weighted}


def build_class_weights(y: np.ndarray) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y)
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


def build_hmm_features(
    labels: pd.DataFrame,
    train_dates: List[pd.Timestamp],
    all_dates: List[pd.Timestamp],
    market_scaler: StandardScaler,
) -> pd.DataFrame:
    n_states = cfg.hmm_components
    cols = [f"hmm_post_{i}" for i in range(n_states)] + [f"hmm_state_{i}" for i in range(n_states)]
    if not HMM_AVAILABLE:
        print("WARNING: hmmlearn not available. Continuing without HMM features.")
        return pd.DataFrame(np.zeros((len(all_dates), 0), dtype=np.float32), index=all_dates)

    X_train = market_scaler.transform(labels.loc[train_dates, MARKET_COLS].values)
    X_all = market_scaler.transform(labels.loc[all_dates, MARKET_COLS].values)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=cfg.hmm_iter,
        min_covar=1e-3,
        random_state=cfg.seed,
        verbose=False,
    )
    hmm.fit(X_train)
    post = hmm.predict_proba(X_all).astype(np.float32)
    states = hmm.predict(X_all).astype(int)
    onehot = np.eye(n_states, dtype=np.float32)[states]
    return pd.DataFrame(np.concatenate([post, onehot], axis=1), index=all_dates, columns=cols)


def fit_node_scaler(returns: pd.DataFrame, labels: pd.DataFrame, train_dates: List[pd.Timestamp]) -> StandardScaler:
    date_to_pos = {d: i for i, d in enumerate(labels.index)}
    rows = []
    for d in train_dates:
        pos = date_to_pos[d]
        ret_1d = returns.iloc[pos].values
        ret_5d = returns.iloc[pos - 4 : pos + 1].sum(axis=0).values
        ret_10d = returns.iloc[pos - 9 : pos + 1].sum(axis=0).values
        vol_30d = returns.iloc[pos - 29 : pos + 1].std(axis=0).values
        excess = ret_1d - returns.iloc[pos].mean()
        rows.append(np.stack([ret_1d, ret_5d, ret_10d, vol_30d, excess], axis=1))
    rows = np.nan_to_num(np.concatenate(rows, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaler.fit(rows)
    return scaler


def main() -> None:
    seed_everything(cfg.seed)
    print("Device:", cfg.device)
    print("Config:", cfg)

    returns, labels = load_data()
    all_dates = make_sample_indices(labels)
    train_dates = [d for d in all_dates if d.year <= 2022]
    val_dates = [d for d in all_dates if d.year == 2023]
    test_dates = [d for d in all_dates if d.year >= 2024]

    print(f"Samples: train {len(train_dates)} val {len(val_dates)} test {len(test_dates)}")
    print("Horizon:", cfg.horizon)

    market_scaler = StandardScaler()
    market_scaler.fit(labels.loc[train_dates, MARKET_COLS].values)
    node_scaler = fit_node_scaler(returns, labels, train_dates)
    hmm_features = build_hmm_features(labels, train_dates, all_dates, market_scaler)
    hmm_in_dim = hmm_features.shape[1]
    if hmm_in_dim > 0:
        print(f"HMM features: {hmm_in_dim} dims ({cfg.hmm_components} posterior + {cfg.hmm_components} state one-hot)")

    train_ds = MarketGraphDataset(returns, labels, train_dates, market_scaler, node_scaler, hmm_features)
    val_ds = MarketGraphDataset(returns, labels, val_dates, market_scaler, node_scaler, hmm_features)
    test_ds = MarketGraphDataset(returns, labels, test_dates, market_scaler, node_scaler, hmm_features)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_graph_sequences)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graph_sequences)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graph_sequences)

    y_train = labels.loc[train_dates, "future_label"].values
    y_val = labels.loc[val_dates, "future_label"].values
    y_test = labels.loc[test_dates, "future_label"].values
    print("Train class counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val class counts:  ", dict(zip(*np.unique(y_val, return_counts=True))))
    print("Test class counts: ", dict(zip(*np.unique(y_test, return_counts=True))))

    weights = build_class_weights(y_train).to(cfg.device)
    print("Class weights:", weights.detach().cpu().numpy())

    model = GATTemporalHMMMarketFusion(node_in_dim=5, market_in_dim=len(MARKET_COLS), hmm_in_dim=hmm_in_dim).to(cfg.device)
    criterion = WeightedFocalLoss(weight=weights, gamma=cfg.focal_gamma, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        yv, pv = predict(model, val_loader, cfg.device)
        val_macro = f1_score(yv, pv, average="macro", zero_division=0)
        val_acc = accuracy_score(yv, pv)
        val_per = f1_score(yv, pv, average=None, labels=[0, 1, 2], zero_division=0)
        scheduler.step(val_macro)
        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | val_acc {val_acc:.4f} | "
            f"val_macro_f1 {val_macro:.4f} | per_class [{val_per[0]:.3f}, {val_per[1]:.3f}, {val_per[2]:.3f}]"
        )

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
    print_eval("UPGRADED GATv2 + HMM + Temporal Attention + Market Fusion (t+3) - Validation", yv, pv)
    print_eval("UPGRADED GATv2 + HMM + Temporal Attention + Market Fusion (t+3) - Test", yt, pt)

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "market_cols": MARKET_COLS,
            "hmm_in_dim": hmm_in_dim,
            "tickers": list(returns.columns),
            "best_val_macro_f1": best_val,
        },
        out_dir / "best_gat_hmm_attention_tplus3.pt",
    )
    print("\nSaved checkpoint: checkpoints/best_gat_hmm_attention_tplus3.pt")


if __name__ == "__main__":
    main()
