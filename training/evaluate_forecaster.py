"""Evaluation suite for regime-conditioned stock forecasters.

This module compares two trained forecasting models:
- StockForecaster with regime conditioning.
- StockForecaster without regime conditioning.

It computes regression and directional metrics, breaks performance down by
market regime and forecast horizon day, and generates visual diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.download import load_ticker
from inference.predict import MarketRegimePredictor
from models.forecaster import (
    StockDataset,
    StockForecaster,
    get_regime_for_dates,
    prepare_stock_data,
)


REGIME_NAMES = {0: "Growth", 1: "Transition", 2: "Panic"}
REGIME_COLORS_LIGHT = {
    0: "rgba(144, 238, 144, 0.18)",
    1: "rgba(255, 165, 0, 0.18)",
    2: "rgba(255, 99, 71, 0.18)",
}


def load_forecaster_models(
    ticker: str = "SPY",
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> tuple[StockForecaster, StockForecaster]:
    """Load both saved forecaster variants for a ticker."""
    checkpoint_dir = Path(checkpoint_dir)

    with_path = checkpoint_dir / f"forecaster_with_regime_{ticker}.pt"
    without_path = checkpoint_dir / f"forecaster_no_regime_{ticker}.pt"

    if not with_path.exists():
        raise FileNotFoundError(f"Missing forecaster checkpoint: {with_path}")
    if not without_path.exists():
        raise FileNotFoundError(f"Missing forecaster checkpoint: {without_path}")

    model_with_regime = StockForecaster(use_regime=True).to(device)
    model_no_regime = StockForecaster(use_regime=False).to(device)

    with_ckpt = torch.load(with_path, map_location=device)
    without_ckpt = torch.load(without_path, map_location=device)

    model_with_regime.load_state_dict(with_ckpt["state_dict"])
    model_no_regime.load_state_dict(without_ckpt["state_dict"])

    model_with_regime.eval()
    model_no_regime.eval()

    return model_with_regime, model_no_regime


def get_test_predictions(
    model_with: StockForecaster,
    model_without: StockForecaster,
    X_test: np.ndarray,
    regimes_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Run both models on test arrays and return aligned prediction payload."""
    x_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    r_tensor = torch.tensor(regimes_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_with = model_with(x_tensor, r_tensor).cpu().numpy()
        pred_without = model_without(x_tensor).cpu().numpy()

    return {
        "pred_with": pred_with,
        "pred_without": pred_without,
        "true": np.asarray(y_test, dtype=float),
        "regimes": np.argmax(regimes_test, axis=1).astype(int),
        "regime_probs": np.asarray(regimes_test, dtype=float),
    }


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy on aggregated move over 5-day horizon."""
    true_dir = np.sign(np.sum(y_true, axis=1))
    pred_dir = np.sign(np.sum(y_pred, axis=1))
    return float(np.mean(true_dir == pred_dir) * 100.0)


def _per_day_directional(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """Directional accuracy per forecast day."""
    out: list[float] = []
    for d in range(y_true.shape[1]):
        td = np.sign(y_true[:, d])
        pdv = np.sign(y_pred[:, d])
        out.append(float(np.mean(td == pdv) * 100.0))
    return out


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def compute_forecaster_metrics(predictions: dict) -> dict:
    """Compute and print comparison metrics for regime/no-regime models."""
    y_true = np.asarray(predictions["true"], dtype=float)
    y_with = np.asarray(predictions["pred_with"], dtype=float)
    y_without = np.asarray(predictions["pred_without"], dtype=float)
    regimes = np.asarray(predictions["regimes"], dtype=int)

    metrics = {
        "with_regime": {
            "mse": float(mean_squared_error(y_true.reshape(-1), y_with.reshape(-1))),
            "mae": float(mean_absolute_error(y_true.reshape(-1), y_with.reshape(-1))),
            "directional_accuracy_overall": _directional_accuracy(y_true, y_with),
        },
        "without_regime": {
            "mse": float(mean_squared_error(y_true.reshape(-1), y_without.reshape(-1))),
            "mae": float(mean_absolute_error(y_true.reshape(-1), y_without.reshape(-1))),
            "directional_accuracy_overall": _directional_accuracy(y_true, y_without),
        },
    }

    per_reg_with: dict[str, float] = {}
    per_reg_without: dict[str, float] = {}

    for rid, rname in REGIME_NAMES.items():
        mask = regimes == rid
        if not np.any(mask):
            per_reg_with[rname] = float("nan")
            per_reg_without[rname] = float("nan")
            continue

        per_reg_with[rname] = _directional_accuracy(y_true[mask], y_with[mask])
        per_reg_without[rname] = _directional_accuracy(y_true[mask], y_without[mask])

    metrics["with_regime"]["directional_accuracy_by_regime"] = per_reg_with
    metrics["without_regime"]["directional_accuracy_by_regime"] = per_reg_without

    day_with = _per_day_directional(y_true, y_with)
    day_without = _per_day_directional(y_true, y_without)
    metrics["with_regime"]["directional_accuracy_by_day"] = {
        f"day_{i+1}": day_with[i] for i in range(len(day_with))
    }
    metrics["without_regime"]["directional_accuracy_by_day"] = {
        f"day_{i+1}": day_without[i] for i in range(len(day_without))
    }

    print("Forecaster Evaluation Results")
    print("=" * 60)
    print("Metric              | With Regime | No Regime | Diff")

    def _line_metric(name: str, vw: float, vn: float, pct: bool = False) -> None:
        diff = vw - vn
        if pct:
            print(f"{name:<19} | {vw:>7.1f}%    | {vn:>7.1f}%   | {diff:+6.1f}%")
        else:
            print(f"{name:<19} | {vw:>9.4f}  | {vn:>8.4f}  | {diff:+7.3f}")

    _line_metric("Overall MSE", metrics["with_regime"]["mse"], metrics["without_regime"]["mse"], pct=False)
    _line_metric("Overall MAE", metrics["with_regime"]["mae"], metrics["without_regime"]["mae"], pct=False)
    _line_metric(
        "Direction (overall)",
        metrics["with_regime"]["directional_accuracy_overall"],
        metrics["without_regime"]["directional_accuracy_overall"],
        pct=True,
    )

    _line_metric(
        "Direction (Growth)",
        per_reg_with["Growth"],
        per_reg_without["Growth"],
        pct=True,
    )
    _line_metric(
        "Direction (Trans)",
        per_reg_with["Transition"],
        per_reg_without["Transition"],
        pct=True,
    )
    _line_metric(
        "Direction (Panic)",
        per_reg_with["Panic"],
        per_reg_without["Panic"],
        pct=True,
    )

    for i in range(5):
        _line_metric(
            f"Day {i+1} accuracy",
            day_with[i],
            day_without[i],
            pct=True,
        )

    print("=" * 60)

    return metrics


def _shade_regime_runs(fig: go.Figure, dates: pd.DatetimeIndex, regimes: np.ndarray, row: int) -> None:
    """Shade contiguous regime periods in a subplot."""
    if len(dates) == 0:
        return

    start = 0
    current = int(regimes[0])

    for i in range(1, len(regimes) + 1):
        changed = i == len(regimes) or int(regimes[i]) != current
        if not changed:
            continue

        x0 = dates[start]
        x1 = dates[i - 1]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=REGIME_COLORS_LIGHT.get(current, "rgba(200,200,200,0.15)"),
            opacity=1.0,
            line_width=0,
            row=row,
            col=1,
        )

        start = i
        if i < len(regimes):
            current = int(regimes[i])


def plot_prediction_vs_actual(
    predictions: dict,
    dates: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot cumulative return tracking and Day-1 prediction error by regime."""
    y_true = np.asarray(predictions["true"], dtype=float)
    y_with = np.asarray(predictions["pred_with"], dtype=float)
    y_without = np.asarray(predictions["pred_without"], dtype=float)
    regimes = np.asarray(predictions["regimes"], dtype=int)
    dt = pd.to_datetime(dates)

    true_day1 = y_true[:, 0]
    with_day1 = y_with[:, 0]
    without_day1 = y_without[:, 0]

    true_cum = np.cumprod(1.0 + true_day1) - 1.0
    with_cum = np.cumprod(1.0 + with_day1) - 1.0
    without_cum = np.cumprod(1.0 + without_day1) - 1.0

    err_with = with_day1 - true_day1
    err_without = without_day1 - true_day1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Cumulative return predictions vs actual",
            "Prediction error by regime period",
        ),
        row_heights=[0.58, 0.42],
    )

    fig.add_trace(
        go.Scatter(x=dt, y=true_cum, mode="lines", name="True cumulative", line=dict(color="black", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=with_cum, mode="lines", name="With regime", line=dict(color="green", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=without_cum, mode="lines", name="Without regime", line=dict(color="red", width=2)),
        row=1,
        col=1,
    )

    _shade_regime_runs(fig, dt, regimes, row=2)

    fig.add_trace(
        go.Scatter(x=dt, y=err_with, mode="lines", name="Error with regime", line=dict(color="green", width=1.7)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=err_without, mode="lines", name="Error without regime", line=dict(color="red", width=1.7)),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="black", dash="dash", width=1), row=2, col=1)

    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Day-1 Error", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(width=1300, height=800, title="Forecaster: predictions vs actual")

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved plot: {save_path}")
        except Exception as exc:
            print(f"Could not save {save_path}: {exc}")

    fig.show()


def plot_regime_performance(
    metrics: dict,
    save_path: Path | None = None,
) -> None:
    """Plot directional accuracy by regime and by forecast day."""
    with_reg = metrics["with_regime"]["directional_accuracy_by_regime"]
    without_reg = metrics["without_regime"]["directional_accuracy_by_regime"]

    reg_names = ["Growth", "Transition", "Panic"]
    with_vals = [with_reg.get(name, np.nan) for name in reg_names]
    without_vals = [without_reg.get(name, np.nan) for name in reg_names]

    with_day = metrics["with_regime"]["directional_accuracy_by_day"]
    without_day = metrics["without_regime"]["directional_accuracy_by_day"]
    days = [1, 2, 3, 4, 5]
    with_day_vals = [with_day[f"day_{d}"] for d in days]
    without_day_vals = [without_day[f"day_{d}"] for d in days]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Directional accuracy by market regime",
            "Accuracy decay over forecast horizon",
        ),
    )

    fig.add_trace(
        go.Bar(x=reg_names, y=with_vals, name="With regime", marker_color="green"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=reg_names, y=without_vals, name="Without regime", marker_color="red"),
        row=1,
        col=1,
    )
    fig.add_hline(y=50.0, line=dict(color="gray", dash="dash", width=1), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=days, y=with_day_vals, mode="lines+markers", name="With regime", line=dict(color="green", width=2)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=days, y=without_day_vals, mode="lines+markers", name="Without regime", line=dict(color="red", width=2)),
        row=1,
        col=2,
    )
    fig.add_hline(y=50.0, line=dict(color="gray", dash="dash", width=1), row=1, col=2)

    fig.update_xaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)

    fig.update_xaxes(title_text="Forecast Day", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

    fig.update_layout(width=1300, height=520, barmode="group", title="Forecaster regime-performance diagnostics")

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved plot: {save_path}")
        except Exception as exc:
            print(f"Could not save {save_path}: {exc}")

    fig.show()


def plot_forecast_sample(
    model_with: StockForecaster,
    ticker: str = "SPY",
    raw_dir: Path = Path("data/raw"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    n_samples: int = 5,
    device: str = "cpu",
) -> None:
    """Visualize random test-window forecasts against realized future paths."""
    model_with.eval()

    ckpt_no = Path(checkpoint_dir) / f"forecaster_no_regime_{ticker}.pt"
    if not ckpt_no.exists():
        raise FileNotFoundError(f"Missing no-regime checkpoint: {ckpt_no}")

    model_without = StockForecaster(use_regime=False).to(device)
    no_ckpt = torch.load(ckpt_no, map_location=device)
    model_without.load_state_dict(no_ckpt["state_dict"])
    model_without.eval()

    predictor = MarketRegimePredictor(device=device)

    X, y, dates = prepare_stock_data(
        ticker=ticker,
        start_date="2000-01-01",
        end_date="2024-12-31",
        raw_dir=raw_dir,
        window_size=30,
        forecast_horizon=5,
    )

    regimes = get_regime_for_dates(dates, predictor)
    dt = pd.to_datetime(dates)
    test_mask = dt >= pd.Timestamp("2023-01-01")

    X_test = X[test_mask]
    y_test = y[test_mask]
    reg_test = regimes[test_mask]
    dates_test = dt[test_mask]

    if len(X_test) == 0:
        raise ValueError("No test samples available for forecast sample plotting.")

    n_pick = min(int(n_samples), len(X_test))
    rng = np.random.default_rng(42)
    pick_idx = rng.choice(len(X_test), size=n_pick, replace=False)

    raw_df = load_ticker(ticker=ticker, save_dir=Path(raw_dir)).sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    close_series = raw_df["Close"].astype(float)

    fig = make_subplots(
        rows=n_pick,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=[f"Sample {i+1}" for i in range(n_pick)],
    )

    for row_num, idx in enumerate(pick_idx, start=1):
        xw = X_test[idx]
        yw = y_test[idx]
        rw = reg_test[idx]
        pred_date = pd.Timestamp(dates_test[idx])

        x_tensor = torch.tensor(xw[None, :, :], dtype=torch.float32, device=device)
        r_tensor = torch.tensor(rw[None, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_with = model_with(x_tensor, r_tensor).cpu().numpy().reshape(-1)
            pred_without = model_without(x_tensor).cpu().numpy().reshape(-1)

        hist_close = close_series.loc[:pred_date].tail(30)
        if hist_close.empty:
            continue

        current_price = float(close_series.loc[pred_date]) if pred_date in close_series.index else float(hist_close.iloc[-1])

        true_prices = []
        with_prices = []
        without_prices = []

        p_true = current_price
        p_with = current_price
        p_without = current_price

        for d in range(5):
            p_true = p_true * (1.0 + float(yw[d]))
            p_with = p_with * (1.0 + float(pred_with[d]))
            p_without = p_without * (1.0 + float(pred_without[d]))

            true_prices.append(p_true)
            with_prices.append(p_with)
            without_prices.append(p_without)

        future_dates = pd.bdate_range(start=pred_date + pd.Timedelta(days=1), periods=5)
        regime_label = REGIME_NAMES.get(int(np.argmax(rw)), "Unknown")

        fig.add_trace(
            go.Scatter(
                x=hist_close.index,
                y=hist_close.values,
                mode="lines",
                name="Historical (30d)" if row_num == 1 else None,
                showlegend=(row_num == 1),
                line=dict(color="gray", width=1.5),
            ),
            row=row_num,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=true_prices,
                mode="lines+markers",
                name="True future" if row_num == 1 else None,
                showlegend=(row_num == 1),
                line=dict(color="black", width=2),
            ),
            row=row_num,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=with_prices,
                mode="lines+markers",
                name="With regime" if row_num == 1 else None,
                showlegend=(row_num == 1),
                line=dict(color="green", width=2, dash="dash"),
            ),
            row=row_num,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=without_prices,
                mode="lines+markers",
                name="Without regime" if row_num == 1 else None,
                showlegend=(row_num == 1),
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=row_num,
            col=1,
        )

        fig.add_annotation(
            xref=f"x{'' if row_num == 1 else row_num}",
            yref=f"y{'' if row_num == 1 else row_num}",
            x=hist_close.index[-1],
            y=float(hist_close.values[-1]),
            text=f"Regime: {regime_label}",
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )

    fig.update_layout(
        height=320 * n_pick,
        width=1200,
        title=f"{ticker} forecast samples: historical vs true vs predicted",
    )
    fig.show()


def run_forecaster_evaluation(
    ticker: str = "SPY",
    processed_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    raw_dir: Path = Path("data/raw"),
    device: str = "cpu",
) -> dict:
    """Run full forecaster evaluation workflow and persist outputs."""
    _ = processed_dir  # Kept for API consistency with pipeline signatures.

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    predictor = MarketRegimePredictor(device=device)

    X, y, dates = prepare_stock_data(
        ticker=ticker,
        start_date="2000-01-01",
        end_date="2024-12-31",
        raw_dir=raw_dir,
        window_size=30,
        forecast_horizon=5,
    )

    regimes = get_regime_for_dates(dates=dates, predictor=predictor)
    date_idx = pd.to_datetime(dates)
    test_mask = date_idx >= pd.Timestamp("2023-01-01")

    X_test = X[test_mask]
    y_test = y[test_mask]
    regimes_test = regimes[test_mask]
    dates_test = date_idx[test_mask].to_numpy()

    if len(X_test) == 0:
        raise ValueError("No test samples found for date split >= 2023-01-01")

    # Constructed for parity with training interfaces and debugging workflows.
    _test_dataset = StockDataset(X_test, y_test, regimes_test)

    model_with, model_without = load_forecaster_models(
        ticker=ticker,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    pred_payload = get_test_predictions(
        model_with=model_with,
        model_without=model_without,
        X_test=X_test,
        regimes_test=regimes_test,
        y_test=y_test,
        device=device,
    )

    metrics = compute_forecaster_metrics(pred_payload)

    plot_prediction_vs_actual(
        predictions=pred_payload,
        dates=dates_test,
        save_path=checkpoint_dir / "forecaster_prediction_vs_actual.png",
    )

    plot_regime_performance(
        metrics=metrics,
        save_path=checkpoint_dir / "forecaster_regime_performance.png",
    )

    plot_forecast_sample(
        model_with=model_with,
        ticker=ticker,
        raw_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        n_samples=5,
        device=device,
    )

    metrics_path = checkpoint_dir / "forecaster_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved metrics: {metrics_path}")

    return metrics


if __name__ == "__main__":
    results = run_forecaster_evaluation()

    improvement = (
        float(results["with_regime"]["directional_accuracy_overall"])
        - float(results["without_regime"]["directional_accuracy_overall"])
    )

    reg_perf = results["with_regime"]["directional_accuracy_by_regime"]
    best_regime = max(reg_perf.items(), key=lambda kv: kv[1] if not np.isnan(kv[1]) else -np.inf)

    print("Forecaster evaluation complete")
    print(f"Key finding: Regime conditioning improves direction accuracy by {improvement:+.1f}%")
    print(f"Best regime for prediction: {best_regime[0]} ({best_regime[1]:.1f}% accuracy)")
    print("Plots saved to models/checkpoints/")
