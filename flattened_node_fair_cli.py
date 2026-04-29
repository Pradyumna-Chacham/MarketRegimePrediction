"""
Fair no-graph / tabular ablation for market regime prediction.
Parameterized by --horizon.

Example:
    python flattened_node_fair_cli.py --horizon 3 --results_dir results_horizon_runs
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results_horizon_runs")
    parser.add_argument("--cache_dir", type=str, default="cache_flat_fair")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


ARGS = parse_args()


@dataclass
class CFG:
    horizon: int = ARGS.horizon
    window: int = 20
    corr_window: int = 20
    top_k: int = 8
    node_dim: int = 5

    batch_size: int = 128
    epochs: int = 20
    patience: int = 5
    lr: float = 8e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.02

    flat_hidden: int = 160
    temporal_hidden: int = 40
    market_hidden: int = 24
    dropout: float = 0.18

    seed: int = ARGS.seed
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()

MARKET_COLS = ["market_return", "market_volatility", "avg_correlation", "cross_sectional_dispersion"]
CACHE_DIR = Path(ARGS.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TAG = f"h{cfg.horizon}_w{cfg.window}_nd{cfg.node_dim}_flatfair"
NODE_FILE = CACHE_DIR / f"node_by_day_{CACHE_TAG}.npy"
TRAIN_CACHE = CACHE_DIR / f"train_split_{CACHE_TAG}.pt"
VAL_CACHE = CACHE_DIR / f"val_split_{CACHE_TAG}.pt"
TEST_CACHE = CACHE_DIR / f"test_split_{CACHE_TAG}.pt"
SCALER_FILE = CACHE_DIR / f"scaler_info_{CACHE_TAG}.npz"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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


def valid_dates(labels: pd.DataFrame) -> List[pd.Timestamp]:
    return list(labels.index[max(cfg.window, 30):])


def split_dates(dates: List[pd.Timestamp]):
    return [d for d in dates if d.year <= 2022], [d for d in dates if d.year == 2023], [d for d in dates if d.year >= 2024]


def node_features_for_day(values: np.ndarray, pos: int) -> np.ndarray:
    r1 = values[pos]
    r5 = values[pos - 4: pos + 1].sum(axis=0)
    r10 = values[pos - 9: pos + 1].sum(axis=0)
    vol30 = values[pos - 29: pos + 1].std(axis=0)
    excess = r1 - np.nanmean(r1)
    x = np.stack([r1, r5, r10, vol30, excess], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_scalers(values: np.ndarray, labels: pd.DataFrame, train_dates: List[pd.Timestamp]):
    date_to_pos = {d: i for i, d in enumerate(labels.index)}
    node_rows = []
    for d in train_dates:
        pos = date_to_pos[d]
        for t in range(pos - cfg.window + 1, pos + 1):
            node_rows.append(node_features_for_day(values, t))
    node_scaler = StandardScaler().fit(np.concatenate(node_rows, axis=0))
    market_scaler = StandardScaler().fit(labels.loc[train_dates, MARKET_COLS].values)
    return node_scaler, market_scaler


def save_scaler_info(node_scaler: StandardScaler, market_scaler: StandardScaler) -> None:
    np.savez(SCALER_FILE, node_mean=node_scaler.mean_, node_scale=node_scaler.scale_, market_mean=market_scaler.mean_, market_scale=market_scaler.scale_)


def scaler_cache_matches(node_scaler: StandardScaler, market_scaler: StandardScaler) -> bool:
    if not SCALER_FILE.exists():
        return False
    try:
        info = np.load(SCALER_FILE)
        return bool(np.allclose(info["node_mean"], node_scaler.mean_) and np.allclose(info["node_scale"], node_scaler.scale_) and np.allclose(info["market_mean"], market_scaler.mean_) and np.allclose(info["market_scale"], market_scaler.scale_))
    except Exception:
        return False


def precompute_all_day_node_features(returns: pd.DataFrame, node_scaler: StandardScaler) -> np.ndarray:
    values = returns.values.astype(np.float32)
    D, N = values.shape
    node_by_day = np.zeros((D, N, cfg.node_dim), dtype=np.float32)
    for pos in tqdm(range(30, D), desc=f"Precomputing flat node tensors h={cfg.horizon}"):
        node_by_day[pos] = node_scaler.transform(node_features_for_day(values, pos)).astype(np.float32)
    return node_by_day


def build_cached_split(labels: pd.DataFrame, dates: List[pd.Timestamp], node_by_day: np.ndarray, market_scaler: StandardScaler) -> Dict[str, torch.Tensor]:
    date_to_pos = {d: i for i, d in enumerate(labels.index)}
    x_seqs, markets, ys = [], [], []
    for d in tqdm(dates, desc="Building fair flattened split", leave=False):
        pos = date_to_pos[d]
        idxs = np.arange(pos - cfg.window + 1, pos + 1)
        x_seqs.append(node_by_day[idxs].reshape(cfg.window, -1))
        market = labels.loc[d, MARKET_COLS].values.reshape(1, -1)
        markets.append(market_scaler.transform(market).astype(np.float32).squeeze(0))
        ys.append(int(labels.loc[d, "future_label"]))
    return {"x": torch.tensor(np.stack(x_seqs), dtype=torch.float32), "market": torch.tensor(np.stack(markets), dtype=torch.float32), "y": torch.tensor(np.array(ys), dtype=torch.long)}


class CachedFlatDataset(Dataset):
    def __init__(self, cache: Dict[str, torch.Tensor]):
        self.cache = cache
    def __len__(self):
        return self.cache["y"].shape[0]
    def __getitem__(self, idx: int):
        return self.cache["x"][idx], self.cache["market"][idx], self.cache["y"][idx]


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, max(dim // 2, 1)), nn.Tanh(), nn.Linear(max(dim // 2, 1), 1))
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (h * w).sum(dim=1)


class FlattenedTableTemporalModel(nn.Module):
    def __init__(self, flat_dim: int, market_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.day_encoder = nn.Sequential(
            nn.Linear(flat_dim, cfg.flat_hidden), nn.LayerNorm(cfg.flat_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.flat_hidden, cfg.temporal_hidden), nn.LayerNorm(cfg.temporal_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.temporal = nn.GRU(input_size=cfg.temporal_hidden, hidden_size=cfg.temporal_hidden, batch_first=True, bidirectional=True)
        self.temporal_attn = TemporalAttention(cfg.temporal_hidden * 2)
        self.market_net = nn.Sequential(nn.Linear(market_dim, cfg.market_hidden), nn.LayerNorm(cfg.market_hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.market_hidden, cfg.market_hidden), nn.GELU())
        fused_dim = cfg.temporal_hidden * 2 + cfg.market_hidden
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(128, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, num_classes))
    def forward(self, x_seq: torch.Tensor, market: torch.Tensor) -> torch.Tensor:
        z = self.day_encoder(x_seq)
        h, _ = self.temporal(z)
        return self.classifier(torch.cat([self.temporal_attn(h), self.market_net(market)], dim=1))


def class_weights(labels: pd.DataFrame, train_dates: List[pd.Timestamp]) -> torch.Tensor:
    y = labels.loc[train_dates, "future_label"].values
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    print("Class counts:", counts.astype(int).tolist())
    print("Class weights:", np.round(weights, 4).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=cfg.device)


def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True if cfg.num_workers > 0 else False)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, desc: str = "Eval") -> Dict:
    model.eval()
    ys, ps = [], []
    for x_seq, market, y in tqdm(loader, desc=desc, leave=False):
        x_seq = x_seq.to(cfg.device, non_blocking=True)
        market = market.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        pred = model(x_seq, market).argmax(dim=1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0), "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0), "f1_calm": per[0], "f1_normal": per[1], "f1_turbulent": per[2], "y_true": y_true, "y_pred": y_pred}


def print_report(name: str, result: Dict) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Accuracy    : {result['accuracy']:.4f}")
    print(f"Macro F1    : {result['macro_f1']:.4f}")
    print(f"Weighted F1 : {result['weighted_f1']:.4f}")
    print(f"Per-class F1: calm={result['f1_calm']:.4f}, normal={result['f1_normal']:.4f}, turbulent={result['f1_turbulent']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(result["y_true"], result["y_pred"], labels=[0, 1, 2]))
    print("\nClassification Report:")
    print(classification_report(result["y_true"], result["y_pred"], labels=[0, 1, 2], zero_division=0))


def load_or_build_caches(returns: pd.DataFrame, labels: pd.DataFrame, train_dates, val_dates, test_dates):
    values = returns.values.astype(np.float32)
    print("\nFitting scalers...")
    node_scaler, market_scaler = fit_scalers(values, labels, train_dates)
    scaler_ok = scaler_cache_matches(node_scaler, market_scaler)
    rebuild_node = (not NODE_FILE.exists()) or (not scaler_ok)
    rebuild_split = (not (TRAIN_CACHE.exists() and VAL_CACHE.exists() and TEST_CACHE.exists())) or rebuild_node
    if not scaler_ok:
        print("Scaler cache mismatch or missing. Rebuilding flattened tensor caches.")
    if rebuild_node:
        node_by_day = precompute_all_day_node_features(returns, node_scaler)
        np.save(NODE_FILE, node_by_day)
        save_scaler_info(node_scaler, market_scaler)
    else:
        print(f"Loading cached flattened node tensors: {NODE_FILE}")
        node_by_day = np.load(NODE_FILE)
    if rebuild_split:
        train_cache = build_cached_split(labels, train_dates, node_by_day, market_scaler)
        val_cache = build_cached_split(labels, val_dates, node_by_day, market_scaler)
        test_cache = build_cached_split(labels, test_dates, node_by_day, market_scaler)
        torch.save(train_cache, TRAIN_CACHE)
        torch.save(val_cache, VAL_CACHE)
        torch.save(test_cache, TEST_CACHE)
    else:
        print("Loading cached train/val/test flattened split tensors...")
        train_cache = torch.load(TRAIN_CACHE, map_location="cpu")
        val_cache = torch.load(VAL_CACHE, map_location="cpu")
        test_cache = torch.load(TEST_CACHE, map_location="cpu")
    return train_cache, val_cache, test_cache


def main() -> None:
    seed_everything(cfg.seed)
    out_dir = Path(ARGS.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", cfg.device)
    print("Config:", cfg)
    returns, labels = load_data()
    dates = valid_dates(labels)
    train_dates, val_dates, test_dates = split_dates(dates)
    print("Returns shape:", returns.shape)
    print("Labels shape :", labels.shape)
    print("Samples: train", len(train_dates), "val", len(val_dates), "test", len(test_dates))
    print("Horizon:", cfg.horizon)
    train_cache, val_cache, test_cache = load_or_build_caches(returns, labels, train_dates, val_dates, test_dates)
    train_loader = make_loader(CachedFlatDataset(train_cache), shuffle=True)
    val_loader = make_loader(CachedFlatDataset(val_cache), shuffle=False)
    test_loader = make_loader(CachedFlatDataset(test_cache), shuffle=False)
    sample_x, sample_market, _ = CachedFlatDataset(train_cache)[0]
    model = FlattenedTableTemporalModel(flat_dim=sample_x.shape[-1], market_dim=sample_market.shape[-1]).to(cfg.device)
    weights = class_weights(labels, train_dates)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    best_val, best_state, wait = -1.0, None, 0
    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for x_seq, market, y in tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False):
            x_seq = x_seq.to(cfg.device, non_blocking=True)
            market = market.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_seq, market), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
        val_result = evaluate(model, val_loader, desc=f"Val {epoch:03d}")
        scheduler.step(val_result["macro_f1"])
        tqdm.write(f"Epoch {epoch:03d} | loss {total_loss / len(train_loader.dataset):.4f} | val_acc {val_result['accuracy']:.4f} | val_macro_f1 {val_result['macro_f1']:.4f} | per_class [{val_result['f1_calm']:.3f}, {val_result['f1_normal']:.3f}, {val_result['f1_turbulent']:.3f}]")
        if val_result["macro_f1"] > best_val + 1e-5:
            best_val = val_result["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                tqdm.write(f"Early stopping at epoch {epoch}. Best val macro F1: {best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    rows = []
    for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        result = evaluate(model, loader, desc=f"Final {split_name}")
        print_report(f"FAIR Flattened Table / No-Graph Temporal Model (t+{cfg.horizon}) - {split_name}", result)
        rows.append({"horizon": cfg.horizon, "model": "Fair Flattened No-Graph", "split": split_name, **{k: v for k, v in result.items() if not k.startswith('y_')}})
    out_path = out_dir / f"flattened_fair_h{cfg.horizon}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
