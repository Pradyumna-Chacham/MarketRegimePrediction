import pandas as pd
from graph_builder_tplus5 import build_graph_for_date


def build_graph_sequence_dataset(
    returns: pd.DataFrame,
    labels_df: pd.DataFrame,
    graph_window: int = 30,
    sequence_length: int = 20,
    corr_threshold: float = 0.5,
    horizon: int = 5,
):
    """
    Build samples where inputs end at date t and target is label at t+horizon.

    Alignment:
      - Graph G_t uses returns[t-29 : t]
      - Sequence uses [G_{t-sequence_length+1}, ..., G_t]
      - Target is labels_df.loc[t+horizon, 'label']
    """
    common_idx = returns.index.intersection(labels_df.index)
    returns = returns.loc[common_idx].copy()
    labels_df = labels_df.loc[common_idx].copy()

    dates = list(returns.index)
    samples = []

    start_i = graph_window - 1 + (sequence_length - 1)
    end_i = len(dates) - horizon

    for i in range(start_i, end_i):
        input_date = dates[i]
        target_date = dates[i + horizon]
        graph_seq = []

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
            "input_date": input_date,
            "target_date": target_date,
            "graphs": graph_seq,
            "label": label,
        })

    return samples


def split_samples_by_date(samples):
    train_samples = [s for s in samples if s["target_date"].year <= 2022]
    val_samples = [s for s in samples if s["target_date"].year == 2023]
    test_samples = [s for s in samples if s["target_date"].year >= 2024]
    return train_samples, val_samples, test_samples


def print_alignment_checks(samples, n=3):
    print("\nAlignment sanity checks:")
    for s in samples[:n]:
        print(
            f"Input end date: {s['input_date']} | Target date: {s['target_date']} | "
            f"label: {s['label']} | num_graphs: {len(s['graphs'])}"
        )
