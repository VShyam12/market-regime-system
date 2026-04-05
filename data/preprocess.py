"""Preprocessing pipeline for normalized feature windows and temporal data splits.

This module transforms engineered features into model-ready arrays by applying
rolling z-score normalization, creating sliding windows, splitting by date,
and saving/loading train/validation/test artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def rolling_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Apply rolling z-score normalization and persist rolling stats."""
    rolling_mean = df.rolling(window=window, min_periods=window).mean()
    rolling_std = df.rolling(window=window, min_periods=window).std()

    zscored = (df - rolling_mean) / rolling_std
    zscored = zscored.replace([np.inf, -np.inf], np.nan)
    zscored = zscored.ffill(limit=5)
    zscored = zscored.dropna()

    stats_dir = Path("data/processed")
    stats_dir.mkdir(parents=True, exist_ok=True)
    rolling_mean.to_parquet(stats_dir / "rolling_mean.parquet")
    rolling_std.to_parquet(stats_dir / "rolling_std.parquet")

    return zscored


def create_windows(df: pd.DataFrame, window_size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and aligned prediction dates from a feature DataFrame."""
    n_rows, n_features = df.shape
    if n_rows <= window_size:
        empty_x = np.empty((0, window_size, n_features), dtype=float)
        empty_dates = np.array([], dtype="datetime64[ns]")
        return empty_x, empty_dates

    X_list: list[np.ndarray] = []
    date_list: list[np.datetime64] = []

    for i in tqdm(range(n_rows - window_size), desc="Creating windows"):
        X_list.append(df.iloc[i : i + window_size].to_numpy(dtype=float))
        # Prediction date is the next timestamp after the input window.
        date_list.append(np.datetime64(df.index[i + window_size]))

    X = np.stack(X_list, axis=0)
    dates = np.array(date_list)
    return X, dates


def split_by_date(
    X: np.ndarray,
    dates: np.ndarray,
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
) -> dict[str, np.ndarray]:
    """Split arrays into train/validation/test segments by date boundaries."""
    if len(X) != len(dates):
        raise ValueError("X and dates must have the same number of samples.")

    date_index = pd.to_datetime(dates)
    train_cutoff = pd.Timestamp(train_end)
    val_cutoff = pd.Timestamp(val_end)

    train_mask = date_index <= train_cutoff
    val_mask = (date_index > train_cutoff) & (date_index <= val_cutoff)
    test_mask = date_index > val_cutoff

    splits: dict[str, np.ndarray] = {
        "X_train": X[train_mask],
        "X_val": X[val_mask],
        "X_test": X[test_mask],
        "dates_train": dates[train_mask],
        "dates_val": dates[val_mask],
        "dates_test": dates[test_mask],
    }

    print(f"Train size: {splits['X_train'].shape[0]}")
    print(f"Validation size: {splits['X_val'].shape[0]}")
    print(f"Test size: {splits['X_test'].shape[0]}")

    return splits


def save_splits(splits: dict[str, np.ndarray], save_dir: Path) -> None:
    """Save split arrays to .npy files in the target directory."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, array in splits.items():
        save_path = save_dir / f"{key}.npy"
        np.save(save_path, array)
        print(f"Saved {key} to {save_path}")


def load_splits(save_dir: Path) -> dict[str, np.ndarray]:
    """Load standard split arrays from .npy files and return them as a dictionary."""
    keys = [
        "X_train",
        "X_val",
        "X_test",
        "dates_train",
        "dates_val",
        "dates_test",
    ]

    loaded: dict[str, np.ndarray] = {}
    for key in keys:
        file_path = save_dir / f"{key}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing split file: {file_path}")

        loaded[key] = np.load(file_path, allow_pickle=False)
        print(f"Loaded {key}: {loaded[key].shape}")

    return loaded


def run_preprocessing(
    processed_dir: Path = Path("data/processed"),
    window_size: int = 60,
    zscore_window: int = 252,
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
) -> dict[str, np.ndarray]:
    """Run end-to-end preprocessing and return split arrays."""
    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    features_df = pd.read_parquet(features_path)
    normalized_df = rolling_zscore(features_df, window=zscore_window)

    X, dates = create_windows(normalized_df, window_size=window_size)
    splits = split_by_date(
        X=X,
        dates=dates,
        train_end=train_end,
        val_end=val_end,
    )

    save_splits(splits=splits, save_dir=processed_dir)
    return splits


if __name__ == "__main__":
    split_data = run_preprocessing()

    x_train = split_data["X_train"]
    x_val = split_data["X_val"]
    x_test = split_data["X_test"]

    print(f"\nX_train shape: {x_train.shape}")
    print(f"X_val shape: {x_val.shape}")
    print(f"X_test shape: {x_test.shape}")

    total_windows = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    if total_windows > 0:
        features_per_timestep = x_train.shape[2] if x_train.size > 0 else (x_val.shape[2] if x_val.size > 0 else x_test.shape[2])
        inferred_window_size = x_train.shape[1] if x_train.size > 0 else (x_val.shape[1] if x_val.size > 0 else x_test.shape[1])
    else:
        features_per_timestep = 0
        inferred_window_size = 0

    print("\nPreprocessing summary:")
    print(f"Total windows: {total_windows}")
    print(f"Features per timestep: {features_per_timestep}")
    print(f"Window size: {inferred_window_size}")
