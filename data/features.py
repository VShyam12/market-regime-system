"""Feature engineering utilities for the Market Regime Detection System.

This module computes technical indicators and cross-asset features from raw
ticker data, then builds and saves a merged feature matrix for modeling.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from data.download import load_ticker


def compute_returns(df: pd.DataFrame, periods: list[int] = [1, 5, 21]) -> pd.DataFrame:
    """Compute percentage returns for the requested periods using Close prices."""
    out = pd.DataFrame(index=df.index)
    for period in periods:
        out[f"return_{period}d"] = df["Close"].pct_change(period)
    return out


def compute_volatility(df: pd.DataFrame, windows: list[int] = [10, 20, 60]) -> pd.DataFrame:
    """Compute annualized rolling volatility from 1-day returns."""
    daily_ret = df["Close"].pct_change(1)
    out = pd.DataFrame(index=df.index)
    for window in windows:
        out[f"vol_{window}d"] = daily_ret.rolling(window).std() * np.sqrt(252)
    return out


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute RSI with Wilder's smoothing."""
    close = df["Close"]
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle zero-loss and flat cases explicitly.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)

    return pd.DataFrame({"rsi_14": rsi}, index=df.index)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, signal, and histogram from Close prices."""
    close = df["Close"]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
        },
        index=df.index,
    )


def compute_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute volume divided by rolling average volume.

    If Volume is missing or all NaN (common for some indices), return ones.
    """
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return pd.DataFrame({"volume_ratio": np.ones(len(df), dtype=float)}, index=df.index)

    rolling_avg = df["Volume"].rolling(window).mean()
    volume_ratio = df["Volume"] / rolling_avg
    return pd.DataFrame({"volume_ratio": volume_ratio}, index=df.index)


def compute_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-specific level, change, and moving-average features."""
    close = vix_df["Close"]
    return pd.DataFrame(
        {
            "vix_level": close,
            "vix_change": close.pct_change(1),
            "vix_ma20": close.rolling(20).mean(),
        },
        index=vix_df.index,
    )


def compute_bond_equity_ratio(spy_df: pd.DataFrame, tlt_df: pd.DataFrame) -> pd.DataFrame:
    """Compute TLT/SPY ratio and its 20-day moving average."""
    ratio = tlt_df["Close"] / spy_df["Close"]
    return pd.DataFrame(
        {
            "bond_equity_ratio": ratio,
            "bond_equity_ma20": ratio.rolling(20).mean(),
        },
        index=ratio.index,
    )


def build_feature_matrix(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Build, clean, save, and return the merged feature matrix."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        spy_df = load_ticker("SPY", raw_dir)
        qqq_df = load_ticker("QQQ", raw_dir)
        iwm_df = load_ticker("IWM", raw_dir)
        vix_df = load_ticker("^VIX", raw_dir)
        tlt_df = load_ticker("TLT", raw_dir)
        xlk_df = load_ticker("XLK", raw_dir)
        xlf_df = load_ticker("XLF", raw_dir)
        xlv_df = load_ticker("XLV", raw_dir)
        xlu_df = load_ticker("XLU", raw_dir)
        xle_df = load_ticker("XLE", raw_dir)
        xli_df = load_ticker("XLI", raw_dir)
        _hyg_df = load_ticker("HYG", raw_dir)
        _gld_df = load_ticker("GLD", raw_dir)
    except FileNotFoundError as exc:
        print(f"Failed to load required raw data: {exc}", file=sys.stderr)
        raise

    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_df in (("SPY", spy_df), ("QQQ", qqq_df), ("IWM", iwm_df)):
        returns = compute_returns(ticker_df)
        vol = compute_volatility(ticker_df)
        rsi = compute_rsi(ticker_df)
        macd = compute_macd(ticker_df)
        vol_ratio = compute_volume_ratio(ticker_df)

        all_feats = pd.concat([returns, vol, rsi, macd, vol_ratio], axis=1)
        all_feats = all_feats.add_prefix(f"{ticker}_")
        feature_frames.append(all_feats)

    feature_frames.append(compute_vix_features(vix_df))
    feature_frames.append(compute_bond_equity_ratio(spy_df, tlt_df))

    for ticker, ticker_df in (
        ("XLK", xlk_df),
        ("XLF", xlf_df),
        ("XLV", xlv_df),
        ("XLU", xlu_df),
        ("XLE", xle_df),
        ("XLI", xli_df),
    ):
        sector_feats = pd.concat(
            [
                compute_returns(ticker_df, periods=[1])[["return_1d"]],
                compute_volatility(ticker_df, windows=[20])[["vol_20d"]],
            ],
            axis=1,
        )
        feature_frames.append(sector_feats.add_prefix(f"{ticker}_"))

    features = pd.concat(feature_frames, axis=1)

    max_nan_ratio = 0.30
    features = features.loc[features.isna().mean(axis=1) <= max_nan_ratio]

    out_path = processed_dir / "features.parquet"
    features.to_parquet(out_path)

    print(f"Feature matrix saved to {out_path}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"First columns: {list(features.columns[:10])}")

    return features


if __name__ == "__main__":
    feature_df = build_feature_matrix(Path("data/raw"), Path("data/processed"))

    print(f"\nResult shape: {feature_df.shape}")
    print("\nNaN values per column:")
    print(feature_df.isna().sum())

    if feature_df.empty:
        print("\nDate range: DataFrame is empty")
    else:
        print(f"\nDate range: {feature_df.index.min()} to {feature_df.index.max()}")
