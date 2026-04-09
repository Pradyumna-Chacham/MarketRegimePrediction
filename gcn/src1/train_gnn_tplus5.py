import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset_builder_tplus5 import (
    build_graph_sequence_dataset,
    split_samples_by_date,
    print_alignment_checks,
)
from model_tplus5 import GCNLSTMClassifier


HORIZON = 5
GRAPH_WINDOW = 30
SEQUENCE_LENGTH = 20
CORR_THRESHOLD = 0.5
BATCH_SIZE = 16
NUM_EPOCHS = 25
LR = 3e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(samples, batch_size=16, shuffle=True):
    idxs = list(range(len(samples)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch_idxs = idxs[i:i + batch_size]
        batch = [samples[j] for j in batch_idxs]
        graph_sequences = [b["graphs"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        yield graph_sequences, labels


def evaluate(model, samples, device, batch_size=16):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for graph_sequences, labels in make_batches(samples, batch_size=batch_size, shuffle=False):
            labels = labels.to(device)
            logits = model(graph_sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(samples)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_labels, all_preds


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    returns = pd.read_csv("../../data/daily_returns.csv", index_col=0, parse_dates=True)
    labels_df = pd.read_csv("../../labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

    samples = build_graph_sequence_dataset(
        returns=returns,
        labels_df=labels_df,
        graph_window=GRAPH_WINDOW,
        sequence_length=SEQUENCE_LENGTH,
        corr_threshold=CORR_THRESHOLD,
        horizon=HORIZON,
    )

    print_alignment_checks(samples, n=3)

    train_samples, val_samples, test_samples = split_samples_by_date(samples)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Test samples:  {len(test_samples)}")

    model = GCNLSTMClassifier(
        node_feat_dim=2,
        gcn_hidden_dim=32,
        graph_emb_dim=32,
        lstm_hidden_dim=64,
        num_classes=3,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for graph_sequences, labels in make_batches(train_samples, batch_size=BATCH_SIZE, shuffle=True):
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(graph_sequences)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_samples)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_samples, device, batch_size=BATCH_SIZE)
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    print("\nBest Val Macro F1:", round(best_val_f1, 4))

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_samples, device, batch_size=BATCH_SIZE)

    print("\n" + "=" * 60)
    print(f"GCN + LSTM (t+{HORIZON}) - Test")
    print("=" * 60)
    print("Test Loss:", round(test_loss, 4))
    print("Accuracy:", round(test_acc, 4))
    print("Macro F1:", round(test_f1, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
