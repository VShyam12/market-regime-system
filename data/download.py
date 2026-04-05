"""Download and load daily market data for all configured regime tickers.

This module provides helpers to download OHLCV price data from yfinance,
persist it as Parquet files under data/raw, and load those files back into
pandas DataFrames for downstream feature engineering and modeling.
"""

from pathlib import Path
import sys
import time

import pandas as pd
from tqdm import tqdm
import yfinance as yf


def _ticker_to_filename(ticker: str) -> str:
    """Convert ticker symbols to safe Parquet filenames."""
    return ticker.replace("^", "") + ".parquet"


def download_ticker(ticker: str, start: str, end: str, save_dir: Path) -> bool:
    """Download one ticker's daily data and save it as a Parquet file."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / _ticker_to_filename(ticker)

        data = yf.download(ticker, start=start, end=end, auto_adjust=True)

        if data is None:
            print(f"No data object returned for {ticker}.", file=sys.stderr)
            return False

        if data.empty:
            print(f"No data returned for {ticker}.", file=sys.stderr)
            return False

        # yfinance may return MultiIndex columns (e.g., with ticker as second level).
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        if "Close" not in data.columns:
            print(
                f"Missing 'Close' column for {ticker}. Available columns: {list(data.columns)}",
                file=sys.stderr,
            )
            return False

        data = data.dropna(subset=["Close"])

        if data.empty:
            print(f"All rows dropped for {ticker} after removing NaN Close values.", file=sys.stderr)
            return False

        data.to_parquet(file_path)
        print(f"Downloaded {ticker}: {len(data)} rows saved to {file_path}")
        return True
    except Exception as exc:
        print(f"Failed to download/save {ticker}: {exc}", file=sys.stderr)
        return False


def download_all(
    tickers: list[str],
    start: str,
    end: str,
    save_dir: Path,
    skip_existing: bool = True,
) -> dict[str, list[str]]:
    """Download all tickers and return success/failure summary lists."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {"success": [], "failed": []}

    for ticker in tqdm(tickers, desc="Downloading tickers"):
        file_path = save_dir / _ticker_to_filename(ticker)

        if skip_existing and file_path.exists():
            print(f"Skipping {ticker}: file already exists at {file_path}")
            results["success"].append(ticker)
            time.sleep(1)
            continue

        if download_ticker(ticker=ticker, start=start, end=end, save_dir=save_dir):
            results["success"].append(ticker)
        else:
            results["failed"].append(ticker)

        time.sleep(1)

    return results


def load_ticker(ticker: str, save_dir: Path) -> pd.DataFrame:
    """Load one saved ticker Parquet file as a DataFrame with DatetimeIndex."""
    file_path = save_dir / _ticker_to_filename(ticker)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file for ticker {ticker} not found at {file_path}. "
            "Run the download step first."
        )

    df = pd.read_parquet(file_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def load_all_tickers(tickers: list[str], save_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all available ticker files and return a ticker -> DataFrame mapping."""
    loaded: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            loaded[ticker] = load_ticker(ticker=ticker, save_dir=save_dir)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}", file=sys.stderr)

    return loaded


if __name__ == "__main__":
    from data.tickers import ALL_TICKERS, START_DATE, END_DATE

    raw_dir = Path("data/raw")

    summary = download_all(
        tickers=ALL_TICKERS,
        start=START_DATE,
        end=END_DATE,
        save_dir=raw_dir,
        skip_existing=True,
    )

    print("\nDownload summary:")
    print(f"Successful: {len(summary['success'])} -> {summary['success']}")
    print(f"Failed: {len(summary['failed'])} -> {summary['failed']}")

    loaded_data = load_all_tickers(ALL_TICKERS, raw_dir)

    print("\nLoaded DataFrame shapes:")
    for ticker, df in loaded_data.items():
        print(f"- {ticker}: {df.shape}")
