import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph_for_date(
    returns_window: pd.DataFrame,
    corr_threshold: float = 0.5,
):
    """
    returns_window: shape [window, num_assets]
    Uses:
      - corr matrix from full window
      - node feature 1 = last-day return
      - node feature 2 = rolling volatility within window
    """
    assets = list(returns_window.columns)
    num_nodes = len(assets)

    corr = returns_window.corr().values
    np.fill_diagonal(corr, 0.0)

    # Node features
    last_return = returns_window.iloc[-1].values
    rolling_vol = returns_window.std(axis=0).values

    x = np.stack([last_return, rolling_vol], axis=1)  # [N, 2]
    x = torch.tensor(x, dtype=torch.float)

    # Build edges from thresholded absolute correlation
    edge_index = []
    edge_weight = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            c = corr[i, j]
            if abs(c) >= corr_threshold:
                edge_index.append([i, j])
                edge_weight.append(abs(c))

    if len(edge_index) == 0:
        # fallback: connect each node to itself if graph is empty
        edge_index = [[i, i] for i in range(num_nodes)]
        edge_weight = [1.0 for _ in range(num_nodes)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    return data