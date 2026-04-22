"""Inference pipeline for market regime prediction using LSTM+BAM+Markov.

This module provides a production-oriented predictor that:
- Loads trained LSTM+BAM checkpoints and Markov parameters.
- Loads raw ticker data from cache or downloads missing data.
- Rebuilds the project feature matrix logic for a requested date interval.
- Applies rolling z-score normalization using precomputed training statistics.
- Generates 60-day windows and performs forward inference.
- Applies VIX-aware probability calibration, Viterbi smoothing, and panic override.
- Returns prediction tables and summary helpers for transitions/current regime.

The main entry points are:
- MarketRegimePredictor
- run_inference
- CLI execution via python -m inference.predict or python inference/predict.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import download_ticker, load_ticker
from data.features import (
    compute_bond_equity_ratio,
    compute_macd,
    compute_returns,
    compute_rsi,
    compute_vix_features,
    compute_volatility,
    compute_volume_ratio,
)
from data.preprocess import create_windows
from data.tickers import ALL_TICKERS, REGIME_LABELS
from models.bam import LSTMBAMModel
from models.markov import load_markov_params, viterbi_decode_with_probs
from training.apply_markov import hybrid_probabilities, rule_based_panic_override


class MarketRegimePredictor:
    """End-to-end market regime predictor with LSTM+BAM and Markov post-processing."""

    def __init__(
        self,
        checkpoint_dir: Path = Path("models/checkpoints"),
        raw_dir: Path = Path("data/raw"),
        processed_dir: Path = Path("data/processed"),
        device: str = "cpu",
    ):
        try:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.raw_dir = Path(raw_dir)
            self.processed_dir = Path(processed_dir)
            self.device = device

            self.raw_dir.mkdir(parents=True, exist_ok=True)

            bam_path = self.checkpoint_dir / "bam_best.pt"
            lstm_path = self.checkpoint_dir / "lstm_best.pt"
            prototypes_path = self.checkpoint_dir / "lstm_prototypes.pt"

            if not bam_path.exists():
                raise FileNotFoundError(f"Missing BAM checkpoint: {bam_path}")
            if not lstm_path.exists():
                raise FileNotFoundError(f"Missing LSTM checkpoint: {lstm_path}")
            if not prototypes_path.exists():
                raise FileNotFoundError(f"Missing prototype checkpoint: {prototypes_path}")

            self.model = LSTMBAMModel(
                lstm_checkpoint=lstm_path,
                prototypes_path=prototypes_path,
                device=self.device,
            )
            bam_state = torch.load(bam_path, map_location=self.device)
            if "bam_state_dict" not in bam_state:
                raise KeyError(f"No 'bam_state_dict' in checkpoint: {bam_path}")
            self.model.bam.load_state_dict(bam_state["bam_state_dict"])
            self.model.eval()

            self.transition_matrix, self.emission_matrix, self.initial_dist = load_markov_params(
                self.checkpoint_dir
            )

            mean_path = self.processed_dir / "rolling_mean.parquet"
            std_path = self.processed_dir / "rolling_std.parquet"
            if not mean_path.exists() or not std_path.exists():
                raise FileNotFoundError(
                    "Missing rolling normalization stats. Expected files: "
                    f"{mean_path} and {std_path}"
                )

            self.rolling_mean = pd.read_parquet(mean_path)
            self.rolling_std = pd.read_parquet(std_path)

            lstm_params = sum(p.numel() for p in self.model.lstm.parameters())
            bam_params = sum(p.numel() for p in self.model.bam.parameters())

            print("MarketRegimePredictor loaded successfully")
            print(f"LSTM params: {lstm_params:,}")
            print(f"BAM params: {bam_params:,}")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MarketRegimePredictor: {exc}") from exc

    def _ensure_raw_data(
        self,
        tickers: list[str],
        fetch_start: str,
        fetch_end: str,
        use_cached: bool,
    ) -> dict[str, pd.DataFrame]:
        """Load ticker data from cache or download missing data."""
        data_map: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                ticker_filename = ticker.replace("^", "") + ".parquet"
                file_path = self.raw_dir / ticker_filename

                if use_cached and file_path.exists():
                    df = load_ticker(ticker=ticker, save_dir=self.raw_dir)
                else:
                    ok = download_ticker(
                        ticker=ticker,
                        start=fetch_start,
                        end=fetch_end,
                        save_dir=self.raw_dir,
                    )
                    if not ok:
                        raise RuntimeError(f"Download failed for {ticker}")
                    df = load_ticker(ticker=ticker, save_dir=self.raw_dir)

                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                data_map[ticker] = df.sort_index()
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare data for ticker {ticker}: {exc}") from exc

        return data_map

    @staticmethod
    def _build_feature_matrix_for_range(data_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Recreate feature engineering logic from data/features.py on provided data."""
        required = [
            "SPY",
            "QQQ",
            "IWM",
            "^VIX",
            "TLT",
            "XLK",
            "XLF",
            "XLV",
            "XLU",
            "XLE",
            "XLI",
            "HYG",
            "GLD",
        ]
        missing = [ticker for ticker in required if ticker not in data_map]
        if missing:
            raise ValueError(
                "Feature generation requires these tickers, but they are missing: "
                + ", ".join(missing)
            )

        spy_df = data_map["SPY"]
        qqq_df = data_map["QQQ"]
        iwm_df = data_map["IWM"]
        vix_df = data_map["^VIX"]
        tlt_df = data_map["TLT"]
        xlk_df = data_map["XLK"]
        xlf_df = data_map["XLF"]
        xlv_df = data_map["XLV"]
        xlu_df = data_map["XLU"]
        xle_df = data_map["XLE"]
        xli_df = data_map["XLI"]

        feature_frames: list[pd.DataFrame] = []

        for ticker, ticker_df in (("SPY", spy_df), ("QQQ", qqq_df), ("IWM", iwm_df)):
            returns = compute_returns(ticker_df)
            vol = compute_volatility(ticker_df)
            rsi = compute_rsi(ticker_df)
            macd = compute_macd(ticker_df)
            vol_ratio = compute_volume_ratio(ticker_df)

            all_feats = pd.concat([returns, vol, rsi, macd, vol_ratio], axis=1)
            feature_frames.append(all_feats.add_prefix(f"{ticker}_"))

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

        features = pd.concat(feature_frames, axis=1).sort_index()

        max_nan_ratio = 0.30
        features = features.loc[features.isna().mean(axis=1) <= max_nan_ratio]
        return features

    def _apply_saved_normalization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score using precomputed rolling mean/std (no recomputation)."""
        common_cols = [
            col
            for col in features_df.columns
            if col in self.rolling_mean.columns and col in self.rolling_std.columns
        ]
        if len(common_cols) != len(features_df.columns):
            missing_cols = sorted(set(features_df.columns) - set(common_cols))
            raise ValueError(
                "Normalization stats missing for feature columns: " + ", ".join(missing_cols)
            )

        ordered = features_df[common_cols]
        aligned_mean = self.rolling_mean[common_cols].reindex(ordered.index).ffill().bfill()
        aligned_std = self.rolling_std[common_cols].reindex(ordered.index).ffill().bfill()

        z = (ordered - aligned_mean) / aligned_std.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        z = z.ffill(limit=5)
        z = z.dropna()
        return z

    def predict(
        self,
        start_date: str,
        end_date: str,
        tickers: list[str] | None = None,
        use_cached: bool = True,
    ) -> pd.DataFrame:
        """Generate regime predictions for a date range.

        Returns a DataFrame with columns:
        date, regime, regime_id, confidence, p_growth, p_transition, p_panic, vix_level
        """
        try:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            if start_ts > end_ts:
                raise ValueError("start_date must be <= end_date")

            selected_tickers = tickers if tickers is not None else ALL_TICKERS

            # Extra lookback supports indicator warm-up and 60-day windows.
            fetch_start_ts = start_ts - pd.Timedelta(days=400)
            fetch_end_ts = end_ts + pd.Timedelta(days=5)

            data_map = self._ensure_raw_data(
                tickers=selected_tickers,
                fetch_start=fetch_start_ts.strftime("%Y-%m-%d"),
                fetch_end=fetch_end_ts.strftime("%Y-%m-%d"),
                use_cached=use_cached,
            )

            features_df = self._build_feature_matrix_for_range(data_map)
            features_df = features_df.loc[(features_df.index >= fetch_start_ts) & (features_df.index <= end_ts)]
            if features_df.empty:
                raise ValueError("Feature matrix is empty for the requested range.")

            normalized = self._apply_saved_normalization(features_df)
            if len(normalized) <= 60:
                raise ValueError(
                    "Not enough normalized rows to create 60-day windows. "
                    f"Need > 60, got {len(normalized)}"
                )

            x_all, window_dates = create_windows(normalized, window_size=60)
            window_dates = pd.to_datetime(window_dates)

            in_range_mask = (window_dates >= start_ts) & (window_dates <= end_ts)
            x_in_range = x_all[in_range_mask]
            pred_dates = window_dates[in_range_mask]

            if len(pred_dates) == 0:
                raise ValueError(
                    "No prediction windows available for the requested date range after warm-up."
                )

            x_tensor = torch.tensor(x_in_range, dtype=torch.float32, device=self.device)

            all_probs: list[np.ndarray] = []
            self.model.eval()
            with torch.no_grad():
                for batch in torch.split(x_tensor, 256):
                    logits, _embedding, _retrieved, _weights = self.model(batch)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)

            bam_probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 3), dtype=float)
            if bam_probs.shape[0] != len(pred_dates):
                raise RuntimeError(
                    "Prediction shape mismatch: "
                    f"probs={bam_probs.shape[0]}, dates={len(pred_dates)}"
                )

            calibrated_probs = hybrid_probabilities(
                probs=bam_probs,
                dates=pred_dates.to_numpy(),
                raw_dir=self.raw_dir,
            )

            smoothed_ids, smoothed_confidence = viterbi_decode_with_probs(
                prob_sequence=calibrated_probs,
                transition_matrix=self.transition_matrix,
                initial_dist=self.initial_dist,
            )

            final_ids = rule_based_panic_override(
                smoothed_predictions=smoothed_ids,
                dates=pred_dates.to_numpy(),
                raw_dir=self.raw_dir,
                vix_panic_threshold=30.0,
                min_consecutive_days=3,
            )

            confidence = smoothed_confidence.astype(float)
            override_mask = final_ids != smoothed_ids
            if np.any(override_mask):
                confidence[override_mask] = np.maximum(
                    confidence[override_mask], calibrated_probs[override_mask, 2]
                )

            vix_level = (
                features_df["vix_level"]
                .reindex(pred_dates)
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )

            label_map = {int(k): str(v) for k, v in REGIME_LABELS.items()}
            regime_names = [label_map.get(int(rid), f"Unknown-{rid}") for rid in final_ids]

            result_df = pd.DataFrame(
                {
                    "date": pred_dates,
                    "regime": regime_names,
                    "regime_id": final_ids.astype(int),
                    "confidence": confidence.astype(float),
                    "p_growth": calibrated_probs[:, 0].astype(float),
                    "p_transition": calibrated_probs[:, 1].astype(float),
                    "p_panic": calibrated_probs[:, 2].astype(float),
                    "vix_level": vix_level,
                }
            )

            return result_df.reset_index(drop=True)
        except Exception as exc:
            raise RuntimeError(
                f"Prediction failed for range {start_date} to {end_date}: {exc}"
            ) from exc

    def predict_latest(self, n_days: int = 252) -> pd.DataFrame:
        """Predict the latest regime sequence over the trailing trading-day window."""
        try:
            if n_days <= 0:
                raise ValueError("n_days must be a positive integer")

            max_data_end = pd.Timestamp("2024-12-31")
            min_data_start = pd.Timestamp("2024-01-01")

            trading_days = pd.bdate_range(end=max_data_end, periods=n_days)
            start_ts = max(trading_days.min(), min_data_start)
            end_ts = max_data_end

            predictions = self.predict(
                start_date=start_ts.strftime("%Y-%m-%d"),
                end_date=end_ts.strftime("%Y-%m-%d"),
            )

            if predictions.empty:
                print("No predictions generated for latest period.")
                return predictions

            latest = predictions.iloc[-1]
            print(
                f"Current regime: {latest['regime']} "
                f"(confidence: {float(latest['confidence']):.1%})"
            )

            return predictions
        except Exception as exc:
            raise RuntimeError(f"predict_latest failed: {exc}") from exc

    def get_regime_transitions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Return all regime transition dates and prior regime durations."""
        try:
            if predictions.empty:
                return pd.DataFrame(
                    columns=["date", "from_regime", "to_regime", "duration_days"]
                )

            required_cols = {"date", "regime", "regime_id"}
            if not required_cols.issubset(predictions.columns):
                missing = sorted(required_cols - set(predictions.columns))
                raise ValueError(f"Missing required prediction columns: {missing}")

            df = predictions.sort_values("date").reset_index(drop=True).copy()

            transitions: list[dict[str, object]] = []
            run_start = 0
            current_id = int(df.loc[0, "regime_id"])

            for i in range(1, len(df)):
                next_id = int(df.loc[i, "regime_id"])
                if next_id != current_id:
                    transitions.append(
                        {
                            "date": pd.to_datetime(df.loc[i, "date"]),
                            "from_regime": str(df.loc[i - 1, "regime"]),
                            "to_regime": str(df.loc[i, "regime"]),
                            "duration_days": int(i - run_start),
                        }
                    )
                    run_start = i
                    current_id = next_id

            return pd.DataFrame(
                transitions,
                columns=["date", "from_regime", "to_regime", "duration_days"],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to compute regime transitions: {exc}") from exc

    def get_current_regime(self, predictions: pd.DataFrame | None = None) -> dict[str, object]:
        """Return a summary dictionary for the current regime state."""
        try:
            if predictions is None:
                predictions = self.predict_latest()

            if predictions.empty:
                raise ValueError("Predictions are empty; cannot compute current regime.")

            df = predictions.sort_values("date").reset_index(drop=True)
            last = df.iloc[-1]
            current_id = int(last["regime_id"])

            start_idx = len(df) - 1
            for i in range(len(df) - 2, -1, -1):
                if int(df.loc[i, "regime_id"]) != current_id:
                    break
                start_idx = i

            since_date = pd.to_datetime(df.loc[start_idx, "date"])
            duration_days = int(len(df) - start_idx)

            return {
                "regime": str(last["regime"]),
                "regime_id": current_id,
                "confidence": float(last["confidence"]),
                "p_growth": float(last["p_growth"]),
                "p_transition": float(last["p_transition"]),
                "p_panic": float(last["p_panic"]),
                "since_date": since_date,
                "duration_days": duration_days,
                "vix_level": float(last["vix_level"])
                if pd.notna(last["vix_level"])
                else np.nan,
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to get current regime: {exc}") from exc


def get_regime_transitions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper to compute regime transitions from predictions.

    Args:
        predictions: Prediction DataFrame with at least date/regime/regime_id columns.

    Returns:
        Transition DataFrame with columns: date, from_regime, to_regime, duration_days.
    """
    try:
        helper = MarketRegimePredictor.__new__(MarketRegimePredictor)
        return MarketRegimePredictor.get_regime_transitions(helper, predictions)
    except Exception as exc:
        raise RuntimeError(f"get_regime_transitions failed: {exc}") from exc


def run_inference(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    save_results: bool = True,
    output_dir: Path = Path("inference"),
) -> pd.DataFrame:
    """Run full inference for a date range and print summary outputs."""
    try:
        predictor = MarketRegimePredictor()

        predictions = predictor.predict(start_date=start_date, end_date=end_date)

        print("\nRegime distribution in results:")
        counts = predictions["regime"].value_counts().reindex(
            ["Growth", "Transition", "Panic"], fill_value=0
        )
        total = max(1, len(predictions))
        for regime_name in ["Growth", "Transition", "Panic"]:
            n_obs = int(counts.loc[regime_name])
            pct = n_obs / total
            print(f"  {regime_name:<10} {n_obs:>5} days ({pct:.1%})")

        transitions = predictor.get_regime_transitions(predictions)
        print("\nRegime transitions found:")
        if transitions.empty:
            print("  None")
        else:
            for _, row in transitions.iterrows():
                dt = pd.to_datetime(row["date"]).date()
                print(
                    f"  {dt}: {row['from_regime']} -> {row['to_regime']} "
                    f"(previous lasted {int(row['duration_days'])} days)"
                )

        current = predictor.get_current_regime(predictions)
        print("\nCurrent regime summary:")
        print(f"  Regime: {current['regime']} (id={current['regime_id']})")
        print(f"  Confidence: {current['confidence']:.1%}")
        print(
            "  Probabilities: "
            f"Growth={current['p_growth']:.3f}, "
            f"Transition={current['p_transition']:.3f}, "
            f"Panic={current['p_panic']:.3f}"
        )
        print(f"  Since: {pd.to_datetime(current['since_date']).date()} ({current['duration_days']} days)")
        print(f"  VIX level: {current['vix_level']:.2f}")

        if save_results:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"predictions_{start_date}_{end_date}.csv"
            predictions.to_csv(out_path, index=False)
            print(f"\nSaved predictions to {out_path}")

        return predictions
    except Exception as exc:
        raise RuntimeError(f"run_inference failed: {exc}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run market regime inference.")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save predictions CSV output")

    args = parser.parse_args()

    try:
        preds = run_inference(
            start_date=args.start,
            end_date=args.end,
            save_results=args.save,
            output_dir=Path("inference"),
        )

        latest = preds.iloc[-1] if not preds.empty else None
        regime_counts = preds["regime"].value_counts().reindex(
            ["Growth", "Transition", "Panic"], fill_value=0
        )
        total_n = max(1, len(preds))

        print("=" * 50)
        print("MARKET REGIME DETECTION SYSTEM")
        print("=" * 50)
        print(f"Period: {args.start} to {args.end}")
        print(f"Predictions: {len(preds)} trading days")

        if latest is not None:
            print(
                f"Current regime: {latest['regime']} "
                f"({float(latest['confidence']):.1%} confidence)"
            )
        else:
            print("Current regime: N/A (no predictions)")

        print("Regime distribution:")
        print(
            f"  Growth:     {int(regime_counts['Growth'])} days "
            f"({int(regime_counts['Growth']) / total_n:.1%})"
        )
        print(
            f"  Transition: {int(regime_counts['Transition'])} days "
            f"({int(regime_counts['Transition']) / total_n:.1%})"
        )
        print(
            f"  Panic:      {int(regime_counts['Panic'])} days "
            f"({int(regime_counts['Panic']) / total_n:.1%})"
        )

        transition_count = 0
        if not preds.empty:
            regime_ids = preds["regime_id"].to_numpy(dtype=int)
            transition_count = int(np.sum(regime_ids[1:] != regime_ids[:-1])) if len(regime_ids) > 1 else 0
        print(f"Regime transitions: {transition_count} detected")
        print("=" * 50)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        raise
