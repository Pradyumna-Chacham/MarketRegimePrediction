# flatnode_baseline.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

HORIZON = 5
WINDOW = 30

returns = pd.read_csv("data/daily_returns.csv", index_col=0, parse_dates=True)
labels_df = pd.read_csv("labeled_data/labeled_data.csv", index_col=0, parse_dates=True)

labels_df['future_label'] = labels_df['label'].shift(-HORIZON)
labels_df = labels_df.dropna(subset=['future_label'])
labels_df['future_label'] = labels_df['future_label'].astype(int)

print(f"Returns shape: {returns.shape}")
print(f"Labels shape:  {labels_df.shape}")

# Build per-stock features at each timestep
records = []
dates = []

for i in range(WINDOW, len(returns) - HORIZON):
    window = returns.iloc[i - WINDOW:i]   # shape (30, 100)
    
    ret_1d  = returns.iloc[i]             # (100,)
    ret_5d  = returns.iloc[i-4:i+1].sum() # (100,)
    vol_30d = window.std()                # (100,)
    
    # Flatten all into one vector: 100 * 3 = 300 features
    row = np.concatenate([ret_1d.values, ret_5d.values, vol_30d.values])
    records.append(row)
    dates.append(returns.index[i])

X = pd.DataFrame(records, index=dates)
y = labels_df.loc[X.index, 'future_label']  # use same future_label column

# Align
X, y = X.align(y, join='inner', axis=0)
X = X.dropna()
y = y.loc[X.index]

# Same chronological split
train_end = pd.Timestamp("2022-12-31")
val_end   = pd.Timestamp("2023-12-31")

X_train = X[X.index <= train_end]
y_train = y[y.index <= train_end]
X_val   = X[(X.index > train_end) & (X.index <= val_end)]
y_val   = y[(y.index > train_end) & (y.index <= val_end)]
X_test  = X[X.index > val_end]
y_test  = y[y.index > val_end]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

for name, model in [("LR Flattened", LogisticRegression(max_iter=1000)),
                    ("SVM Flattened", SVC(kernel='rbf'))]:
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    print(f"{name} | Test Acc: {accuracy_score(y_test, preds):.4f} | Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
    print(classification_report(y_test, preds))