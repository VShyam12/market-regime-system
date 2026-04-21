"""Rule-based market regime labeling.

Labeling rules use SPY trend/drawdown and VIX stress as follows:
- Panic (2): VIX > 30, or deep drawdown, or strongly negative 20-day return
- Growth (0): low VIX, shallow drawdown, and positive 20-day return
- Transition (1): all other states

After raw labels are assigned, a run-length smoothing pass reduces short-lived
regime flips by replacing short runs with surrounding majority regimes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


def compute_drawdown(prices: pd.Series) -> pd.Series:
    """Compute percentage drawdown from a 252-day rolling peak."""
    rolling_max = prices.rolling(window=252, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


def assign_regime_rules(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.Series:
    """Assign raw regime labels from SPY and VIX signals.

    Priority order:
    1) Panic
    2) Growth
    3) Transition
    """
    spy_close = spy_df["Close"]
    vix_level = vix_df["Close"]

    drawdown_252 = compute_drawdown(spy_close)
    return_20d = spy_close.pct_change(20)
    return_1d = spy_close.pct_change(1)
    _ = return_1d  # Explicitly computed by requirement; available for future diagnostics.

    labels = pd.Series(1, index=spy_close.index, dtype="int64")

    panic_mask = (vix_level > 30) | (drawdown_252 < -0.20) | (return_20d < -0.10)
    growth_mask = (vix_level < 20) & (drawdown_252 > -0.05) & (return_20d > 0)

    labels.loc[panic_mask.fillna(False)] = 2
    labels.loc[growth_mask.fillna(False) & (~panic_mask.fillna(False))] = 0

    return labels.astype(int)


def smooth_labels(labels: pd.Series, min_duration: int = 5) -> pd.Series:
    """Smooth short label runs by replacing them with surrounding majority labels.

    The process is repeated 3 times to collapse nested short runs.
    """
    smoothed = labels.astype(int).copy()

    for _ in range(3):
        values = smoothed.to_numpy(copy=True)
        n = len(values)
        if n == 0:
            return smoothed

        runs: list[tuple[int, int, int, int]] = []  # (start, end, label, length)
        start = 0
        while start < n:
            end = start
            while end + 1 < n and values[end + 1] == values[start]:
                end += 1
            runs.append((start, end, int(values[start]), end - start + 1))
            start = end + 1

        updated = values.copy()

        for run_idx, (run_start, run_end, run_label, run_len) in enumerate(runs):
            if run_len >= min_duration:
                continue

            prev_label = runs[run_idx - 1][2] if run_idx > 0 else None
            prev_len = runs[run_idx - 1][3] if run_idx > 0 else 0
            next_label = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else None
            next_len = runs[run_idx + 1][3] if run_idx + 1 < len(runs) else 0

            replacement = run_label
            if prev_label is not None and next_label is not None:
                if prev_label == next_label:
                    replacement = prev_label
                else:
                    replacement = prev_label if prev_len >= next_len else next_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label

            updated[run_start : run_end + 1] = replacement

        smoothed = pd.Series(updated, index=smoothed.index, dtype="int64")

    return smoothed.astype(int)


def build_labels(raw_dir: Path, labels_dir: Path) -> pd.Series:
    """Build, smooth, save, and report regime labels."""
    from data.download import load_ticker

    labels_dir.mkdir(parents=True, exist_ok=True)

    spy_df = load_ticker("SPY", raw_dir)
    vix_df = load_ticker("^VIX", raw_dir)

    aligned = pd.concat(
        [
            spy_df[["Close"]].rename(columns={"Close": "SPY_Close"}),
            vix_df[["Close"]].rename(columns={"Close": "VIX_Close"}),
        ],
        axis=1,
        join="inner",
    ).dropna()

    aligned_spy = pd.DataFrame({"Close": aligned["SPY_Close"]}, index=aligned.index)
    aligned_vix = pd.DataFrame({"Close": aligned["VIX_Close"]}, index=aligned.index)

    raw_labels = assign_regime_rules(aligned_spy, aligned_vix)
    smoothed = smooth_labels(raw_labels, min_duration=5)

    labels_path = labels_dir / "regime_labels.parquet"
    smoothed.rename("regime").to_frame().to_parquet(labels_path)

    counts = smoothed.value_counts().sort_index()
    summary = pd.DataFrame({"label": counts.index, "count": counts.values})
    summary["percentage"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    summary_path = labels_dir / "label_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved labels to: {labels_path}")
    print(f"Saved summary to: {summary_path}")
    print("Label distribution:")
    for _, row in summary.iterrows():
        print(f"- Label {int(row['label'])}: {int(row['count'])} days ({row['percentage']:.2f}%)")

    return smoothed


def load_labels(labels_dir: Path) -> pd.Series:
    """Load saved regime labels from Parquet."""
    labels_path = labels_dir / "regime_labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Regime label file not found at {labels_path}. "
            "Run build_labels(...) first to generate labels."
        )

    loaded = pd.read_parquet(labels_path)
    if "regime" in loaded.columns:
        series = loaded["regime"]
    elif loaded.shape[1] == 1:
        series = loaded.iloc[:, 0]
    else:
        raise ValueError(
            f"Expected a 'regime' column in {labels_path}, found columns: {list(loaded.columns)}"
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    return series.astype(int)


def align_labels_with_features(
    labels: pd.Series, features_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Align labels and features on shared dates."""
    common_dates = features_df.index.intersection(labels.index)
    aligned_features = features_df.loc[common_dates].copy()
    aligned_labels = labels.loc[common_dates].copy()

    print(f"Aligned features/labels on {len(common_dates)} shared dates")
    return aligned_features, aligned_labels


if __name__ == "__main__":
    try:
        labels = build_labels(Path("data/raw"), Path("data/labels"))

        features_path = Path("data/processed/features.parquet")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found at {features_path}")

        features_df = pd.read_parquet(features_path)
        aligned_features, aligned_labels = align_labels_with_features(labels, features_df)

        total_days = len(labels)
        distribution = labels.value_counts().sort_index()
        label_names = {0: "Growth", 1: "Transition", 2: "Panic"}

        print(f"\nTotal labelled days: {total_days}")
        print("Label distribution:")
        for label_value in [0, 1, 2]:
            count = int(distribution.get(label_value, 0))
            pct = (count / total_days * 100) if total_days > 0 else 0.0
            print(f"- {label_names[label_value]} {count} days ({pct:.2f}%)")

        if total_days > 0:
            print(f"Date range of labels: {labels.index.min()} to {labels.index.max()}")
        else:
            print("Date range of labels: no labels available")

        print(f"Number of aligned feature-label pairs: {len(aligned_labels)}")

    except Exception as exc:
        print(f"Label pipeline failed: {exc}", file=sys.stderr)
        raise
