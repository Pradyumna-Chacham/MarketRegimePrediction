import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph_for_date(
    returns_window: pd.DataFrame,
    corr_threshold: float = 0.5,
):
    """
    Graph from a rolling returns window.

    Node features:
      - last-day return
      - rolling volatility in the window

    Features are normalized within each graph for stability.
    """
    num_nodes = returns_window.shape[1]

    corr = returns_window.corr().values.astype(np.float32)
    np.fill_diagonal(corr, 0.0)

    last_return = returns_window.iloc[-1].values.astype(np.float32)
    rolling_vol = returns_window.std(axis=0).values.astype(np.float32)

    eps = 1e-8
    last_return = (last_return - last_return.mean()) / (last_return.std() + eps)
    rolling_vol = (rolling_vol - rolling_vol.mean()) / (rolling_vol.std() + eps)

    x = np.stack([last_return, rolling_vol], axis=1)
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            c = corr[i, j]
            if abs(c) >= corr_threshold:
                edge_index.append([i, j])
                edge_weight.append(abs(float(c)))

    if not edge_index:
        edge_index = [[i, i] for i in range(num_nodes)]
        edge_weight = [1.0 for _ in range(num_nodes)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
