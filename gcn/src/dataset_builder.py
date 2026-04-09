import pandas as pd
from typing import List, Tuple

from graph_builder import build_graph_for_date


def build_graph_sequence_dataset(
    returns: pd.DataFrame,
    labels_df: pd.DataFrame,
    graph_window: int = 30,
    sequence_length: int = 20,
    corr_threshold: float = 0.5,
):
    """
    Returns a list of samples:
      sample = {
         "date": target_date,
         "graphs": [Data, Data, ..., Data]  # len = sequence_length
         "label": int
      }

    Alignment:
      Graph G_t uses returns[t-29 : t]
      Label y_t is labels_df.loc[t]
      Sequence uses [G_{t-19}, ..., G_t]
    """
    common_idx = returns.index.intersection(labels_df.index)
    returns = returns.loc[common_idx].copy()
    labels_df = labels_df.loc[common_idx].copy()

    dates = list(returns.index)
    samples = []

    # Need enough history for graph window and sequence length
    start_i = graph_window - 1 + (sequence_length - 1)

    for i in range(start_i, len(dates)):
        graph_seq = []
        target_date = dates[i]

        # sequence ends at i, starts at i-seq+1
        for seq_i in range(i - sequence_length + 1, i + 1):
            w_start = seq_i - graph_window + 1
            w_end = seq_i + 1
            returns_window = returns.iloc[w_start:w_end]

            graph = build_graph_for_date(
                returns_window=returns_window,
                corr_threshold=corr_threshold,
            )
            graph_seq.append(graph)

        label = int(labels_df.loc[target_date, "label"])

        samples.append({
            "date": target_date,
            "graphs": graph_seq,
            "label": label,
        })

    return samples


def split_samples_by_date(samples):
    train_samples = [s for s in samples if s["date"].year <= 2022]
    val_samples = [s for s in samples if s["date"].year == 2023]
    test_samples = [s for s in samples if s["date"].year >= 2024]
    return train_samples, val_samples, test_samples


def print_alignment_checks(samples, n=3):
    print("\nAlignment sanity checks:")
    for s in samples[:n]:
        print(f"Target date: {s['date']}, label: {s['label']}, num_graphs: {len(s['graphs'])}")