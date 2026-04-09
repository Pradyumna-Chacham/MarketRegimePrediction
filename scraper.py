from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

START_DATE = "2017-01-01"
END_DATE = "2026-03-31"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "ADBE", "CRM","AMD", "QCOM", "INTC", "TXN", "AVGO",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "SPGI", "ICE", "CME",
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "TMO", "BMY", "DHR", "MDT", "AMGN", "GILD",
    "CAT", "HON", "UPS", "BA", "DE", "GE", "UNP", "LMT", "RTX", "NOC", "FDX", "EMR",
    "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "KMB", "GIS", "HSY",
    "MCD", "NKE", "SBUX", "HD", "LOW", "TGT", "BKNG", "TJX", "F", "GM",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC",
    "DIS", "CMCSA", "NFLX", "TMUS", "VZ", "T",
    "NEE", "DUK", "SO", "D", "EXC",
    "LIN", "APD", "SHW", "FCX", "NEM",
    "V", "MA", "PYPL", "ADP",
    # stitched Block series will be saved as SQ
    "SQ", "XYZ",
]

CANONICAL_NAME_MAP = {
    "XYZ": "SQ",  # merge XYZ into SQ
}

def download_adjusted_close(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        progress=True,
        threads=True,
    )

    if df.empty:
        raise ValueError("No data returned from yfinance.")

    prices = df["Adj Close"].copy()
    prices.columns.name = None
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def stitch_sq_xyz(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create one continuous 'SQ' series using:
    - old SQ prices before the ticker change
    - XYZ prices where SQ is unavailable
    Then drop XYZ.
    """
    sq = prices["SQ"] if "SQ" in prices.columns else pd.Series(index=prices.index, dtype=float)
    xyz = prices["XYZ"] if "XYZ" in prices.columns else pd.Series(index=prices.index, dtype=float)

    # Prefer SQ where available; otherwise use XYZ
    merged_sq = sq.combine_first(xyz)

    prices = prices.copy()
    prices["SQ"] = merged_sq

    if "XYZ" in prices.columns:
        prices = prices.drop(columns=["XYZ"])

    return prices


def summarize_missingness(prices: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "ticker": prices.columns,
        "missing_count": prices.isna().sum().values,
        "missing_pct": (prices.isna().mean() * 100).values,
        "non_null_count": prices.notna().sum().values,
    }).sort_values(["missing_pct", "ticker"], ascending=[False, True])
    return summary


def clean_price_matrix(prices: pd.DataFrame, max_missing_pct: float = 5.0) -> pd.DataFrame:
    missing = summarize_missingness(prices)
    keep = missing.loc[missing["missing_pct"] <= max_missing_pct, "ticker"].tolist()

    cleaned = prices[keep].copy()
    cleaned = cleaned.ffill()
    cleaned = cleaned.dropna(axis=0, how="any")
    return cleaned


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def main() -> None:
    prices = download_adjusted_close(TICKERS, START_DATE, END_DATE)
    print("Raw downloaded shape:", prices.shape)

    if "SQ" in prices.columns or "XYZ" in prices.columns:
        print("\nBefore stitching:")
        print(prices[["SQ", "XYZ"]].isna().sum())

    prices = stitch_sq_xyz(prices)

    print("\nAfter stitching SQ/XYZ:")
    if "SQ" in prices.columns:
        print("SQ missing count:", int(prices["SQ"].isna().sum()))
    print("Shape after stitching:", prices.shape)

    raw_missingness = summarize_missingness(prices)
    print("\nTop tickers by missingness:")
    print(raw_missingness.head(10))

    cleaned_prices = clean_price_matrix(prices, max_missing_pct=5.0)
    returns = compute_daily_returns(cleaned_prices)

    cleaned_missingness = summarize_missingness(cleaned_prices)

    prices.to_csv(OUTPUT_DIR / "raw_prices.csv")
    raw_missingness.to_csv(OUTPUT_DIR / "raw_missingness.csv", index=False)
    cleaned_prices.to_csv(OUTPUT_DIR / "clean_prices.csv")
    cleaned_missingness.to_csv(OUTPUT_DIR / "clean_missingness.csv", index=False)
    returns.to_csv(OUTPUT_DIR / "daily_returns.csv")

    print("\nSaved:")
    print(OUTPUT_DIR / "raw_prices.csv")
    print(OUTPUT_DIR / "raw_missingness.csv")
    print(OUTPUT_DIR / "clean_prices.csv")
    print(OUTPUT_DIR / "clean_missingness.csv")
    print(OUTPUT_DIR / "daily_returns.csv")


if __name__ == "__main__":
    main()