# turbulent_event_windows.py

import pandas as pd

PATH = "labeled_data/labeled_data.csv"

df = pd.read_csv(PATH, index_col=0, parse_dates=True)

# Keep only turbulent days
turb = df[df["label"] == 2].copy()

# Build consecutive turbulent windows
windows = []
if len(turb) > 0:
    dates = turb.index.sort_values()
    start = dates[0]
    prev = dates[0]

    for d in dates[1:]:
        # market days are not always 1 calendar day apart, so allow weekend gaps
        if (d - prev).days <= 4:
            prev = d
        else:
            windows.append((start, prev))
            start = d
            prev = d

    windows.append((start, prev))

# Summarize each turbulent window
rows = []
for start, end in windows:
    w = df.loc[start:end].copy()

    rows.append({
        "start": start.date(),
        "end": end.date(),
        "calendar_days": (end - start).days + 1,
        "trading_days": len(w),
        "max_regime_score": w["regime_score"].max(),
        "mean_regime_score": w["regime_score"].mean(),
        "max_market_volatility": w["market_volatility"].max(),
        "mean_market_volatility": w["market_volatility"].mean(),
        "max_avg_correlation": w["avg_correlation"].max(),
        "mean_avg_correlation": w["avg_correlation"].mean(),
        "max_cross_sectional_dispersion": w["cross_sectional_dispersion"].max(),
        "mean_cross_sectional_dispersion": w["cross_sectional_dispersion"].mean(),
    })

summary = pd.DataFrame(rows)

# Sort by start date
summary = summary.sort_values("start").reset_index(drop=True)

print("\n================ TURBULENT WINDOWS ================\n")
print(summary.to_string(index=False))

print("\n================ COPY-PASTE CSV VERSION ================\n")
print(summary.to_csv(index=False))

summary.to_csv("turbulent_windows_summary.csv", index=False)
print("\nSaved: turbulent_windows_summary.csv")