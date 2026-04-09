import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch


class GCNEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch = data.batch

        graph_emb = global_mean_pool(x, batch)
        return graph_emb


class GCNLSTMClassifier(nn.Module):
    def __init__(
        self,
        node_feat_dim=2,
        gcn_hidden_dim=32,
        graph_emb_dim=32,
        lstm_hidden_dim=64,
        num_classes=3,
        dropout=0.2,
    ):
        super().__init__()
        self.gcn = GCNEncoder(
            in_channels=node_feat_dim,
            hidden_dim=gcn_hidden_dim,
            out_dim=graph_emb_dim,
        )
        self.lstm = nn.LSTM(
            input_size=graph_emb_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, graph_sequence_batch):
        """
        graph_sequence_batch: list of length B
          each item is a list of torch_geometric Data objects of length T
        """
        batch_size = len(graph_sequence_batch)
        seq_len = len(graph_sequence_batch[0])

        sequence_embeddings = []

        for t in range(seq_len):
            graphs_at_t = [graph_sequence_batch[b][t] for b in range(batch_size)]
            batch_graph = Batch.from_data_list(graphs_at_t)
            graph_emb = self.gcn(batch_graph)  # [B, graph_emb_dim]
            sequence_embeddings.append(graph_emb)

        seq_tensor = torch.stack(sequence_embeddings, dim=1)  # [B, T, D]
        lstm_out, _ = self.lstm(seq_tensor)
        final_hidden = lstm_out[:, -1, :]
        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits