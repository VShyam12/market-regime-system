"""Regime-conditioned stock forecasting with dual-model comparison.

This module implements a BiLSTM forecaster that can optionally condition on
market regime probabilities from the regime detection pipeline. It supports:
- Feature/target preparation from OHLCV time series.
- Regime probability alignment for training dates.
- Training and evaluation of two models (with and without regime inputs).
- Comparison metrics (MSE, MAE, directional accuracy).
- Five-step forecast generation with simple confidence bounds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.download import load_ticker
from inference.predict import MarketRegimePredictor


class StockForecaster(nn.Module):
    """BiLSTM forecaster with optional regime conditioning."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        regime_dim: int = 3,
        forecast_horizon: int = 5,
        use_regime: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.regime_dim = regime_dim
        self.forecast_horizon = forecast_horizon
        self.use_regime = use_regime

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        if self.use_regime:
            self.regime_projection = nn.Sequential(
                nn.Linear(regime_dim, 8),
                nn.ReLU(),
            )
            head_input_dim = hidden_size * 2 + 8
        else:
            self.regime_projection = None
            head_input_dim = hidden_size * 2

        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor, regime: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            regime: Optional tensor of shape (batch, 3).

        Returns:
            Forecast tensor of shape (batch, forecast_horizon).
        """
        out, _ = self.lstm(x)
        lstm_out = out[:, -1, :]

        if self.use_regime and regime is not None:
            regime_proj = self.regime_projection(regime)
            combined = torch.cat([lstm_out, regime_proj], dim=-1)
        else:
            combined = lstm_out

        preds = self.prediction_head(combined)
        return preds


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using exponentially weighted gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the six requested daily forecasting features."""
    close = df["Close"].astype(float)

    feats = pd.DataFrame(index=df.index)
    feats["return_1d"] = close.pct_change(1)
    feats["return_5d"] = close.pct_change(5)
    feats["vol_10d"] = close.pct_change(1).rolling(10).std() * np.sqrt(252)
    feats["rsi_14"] = _compute_rsi(close, window=14)

    if "Volume" in df.columns and not df["Volume"].isna().all():
        feats["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    else:
        feats["volume_ratio"] = 1.0

    close_ma20 = close.rolling(20).mean()
    close_std20 = close.rolling(20).std()
    feats["price_norm"] = (close - close_ma20) / close_std20.replace(0, np.nan)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


def prepare_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    window_size: int = 30,
    forecast_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature windows and multi-step return targets.

    Returns:
        X: (n_samples, window_size, 6)
        y: (n_samples, forecast_horizon)
        dates: prediction dates aligned to each sample
    """
    raw_dir = Path(raw_dir)
    df = load_ticker(ticker=ticker, save_dir=raw_dir).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Extra lookback supports rolling features and window construction.
    pad_start = start_ts - pd.Timedelta(days=180)
    df = df.loc[(df.index >= pad_start) & (df.index <= end_ts + pd.Timedelta(days=forecast_horizon + 5))].copy()

    if df.empty:
        raise ValueError(f"No data found for {ticker} in requested period.")

    feats = _feature_frame(df)
    close = df["Close"].astype(float)

    features_arr = feats.to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    date_arr = df.index.to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    d_list: list[np.datetime64] = []

    n_rows = len(df)
    for i in range(window_size, n_rows - forecast_horizon):
        x_window = features_arr[i - window_size : i, :]

        # Next-5 daily returns from t+1 to t+forecast_horizon.
        step_returns = []
        for h in range(1, forecast_horizon + 1):
            prev_close = close_arr[i + h - 1]
            next_close = close_arr[i + h]
            step_returns.append((next_close - prev_close) / prev_close)
        y_vec = np.asarray(step_returns, dtype=float)

        if np.isnan(x_window).any() or np.isnan(y_vec).any():
            continue

        pred_date = pd.Timestamp(date_arr[i])
        if pred_date < start_ts or pred_date > end_ts:
            continue

        X_list.append(x_window)
        y_list.append(y_vec)
        d_list.append(np.datetime64(pred_date))

    if not X_list:
        raise ValueError("No valid samples after feature/target preparation.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    dates = np.asarray(d_list)

    return X, y, dates


def get_regime_for_dates(dates: np.ndarray, predictor: object) -> np.ndarray:
    """Map dates to regime probabilities using nearest available prediction date."""
    if dates.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    date_idx = pd.to_datetime(dates)

    max_supported_date = pd.Timestamp("2024-12-31")
    over_mask = date_idx > max_supported_date
    if bool(over_mask.any()):
        print(
            f"Warning: {int(over_mask.sum())} date(s) exceed 2024-12-31; capping to 2024-12-31 for regime lookup."
        )
        date_idx = date_idx.where(~over_mask, max_supported_date)

    start = (date_idx.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (date_idx.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    pred_df = predictor.predict(start_date=start, end_date=end)
    if pred_df.empty:
        raise ValueError("Predictor returned no regime predictions for requested dates.")

    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date").reset_index(drop=True)

    pred_dates = pred_df["date"].to_numpy(dtype="datetime64[ns]")
    prob_cols = ["p_growth", "p_transition", "p_panic"]
    probs = pred_df[prob_cols].to_numpy(dtype=np.float32)

    out = np.zeros((len(date_idx), 3), dtype=np.float32)

    for i, dt in enumerate(date_idx.to_numpy(dtype="datetime64[ns]")):
        insert_pos = int(np.searchsorted(pred_dates, dt, side="left"))

        if insert_pos <= 0:
            nearest = 0
        elif insert_pos >= len(pred_dates):
            nearest = len(pred_dates) - 1
        else:
            left = pred_dates[insert_pos - 1]
            right = pred_dates[insert_pos]
            nearest = insert_pos - 1 if abs(dt - left) <= abs(right - dt) else insert_pos

        out[i] = probs[nearest]

    return out


class StockDataset(Dataset):
    """Torch dataset for stock windows, regime probabilities, and targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray, regimes: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.regimes = torch.tensor(regimes, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.regimes[idx], self.y[idx]


def _train_one_model(
    model: StockForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_regime: bool,
    num_epochs: int,
    learning_rate: float,
    device: str,
    patience: int = 10,
) -> tuple[StockForecaster, dict[str, list[float]]]:
    """Train one forecaster model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state: dict | None = None
    best_val = float("inf")
    bad_epochs = 0

    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(num_epochs):
        model.train()
        train_losses: list[float] = []

        for xb, rb, yb in train_loader:
            xb = xb.to(device)
            rb = rb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb, rb) if use_regime else model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, rb, yb in val_loader:
                xb = xb.to(device)
                rb = rb.to(device)
                yb = yb.to(device)

                preds = model(xb, rb) if use_regime else model(xb)
                loss = criterion(preds, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def _evaluate_model(
    model: StockForecaster,
    test_loader: DataLoader,
    use_regime: bool,
    device: str,
) -> dict[str, float]:
    """Evaluate regression and directional metrics on test set."""
    model.eval()

    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []

    with torch.no_grad():
        for xb, rb, yb in test_loader:
            xb = xb.to(device)
            rb = rb.to(device)

            preds = model(xb, rb) if use_regime else model(xb)
            pred_list.append(preds.cpu().numpy())
            true_list.append(yb.numpy())

    if not pred_list:
        return {"mse": np.nan, "mae": np.nan, "directional_accuracy": np.nan}

    y_pred = np.concatenate(pred_list, axis=0)
    y_true = np.concatenate(true_list, axis=0)

    mse = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))

    pred_dir = np.sign(y_pred.sum(axis=1))
    true_dir = np.sign(y_true.sum(axis=1))
    directional_acc = float((pred_dir == true_dir).mean() * 100.0)

    return {
        "mse": mse,
        "mae": mae,
        "directional_accuracy": directional_acc,
    }


def train_forecaster(
    ticker: str = "SPY",
    raw_dir: Path = Path("data/raw"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_regime: bool = True,
    device: str = "cpu",
) -> dict:
    """Train and compare regime-conditioned vs non-conditioned stock forecasters."""
    torch.manual_seed(42)
    np.random.seed(42)

    _ = use_regime  # Both variants are trained by design in this function.

    X, y, dates = prepare_stock_data(
        ticker=ticker,
        start_date="2000-01-01",
        end_date="2024-12-31",
        raw_dir=raw_dir,
        window_size=30,
        forecast_horizon=5,
    )

    predictor = MarketRegimePredictor(device=device)
    regimes = get_regime_for_dates(dates=dates, predictor=predictor)

    date_idx = pd.to_datetime(dates)
    train_mask = date_idx < pd.Timestamp("2021-01-01")
    val_mask = (date_idx >= pd.Timestamp("2021-01-01")) & (date_idx <= pd.Timestamp("2022-12-31"))
    test_mask = date_idx >= pd.Timestamp("2023-01-01")

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Date split produced an empty train/val/test segment.")

    print(
        f"Train: {int(train_mask.sum())} samples, "
        f"Val: {int(val_mask.sum())} samples, "
        f"Test: {int(test_mask.sum())} samples"
    )

    train_ds = StockDataset(X[train_mask], y[train_mask], regimes[train_mask])
    val_ds = StockDataset(X[val_mask], y[val_mask], regimes[val_mask])
    test_ds = StockDataset(X[test_mask], y[test_mask], regimes[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model_with_regime = StockForecaster(use_regime=True).to(device)
    model_no_regime = StockForecaster(use_regime=False).to(device)

    model_with_regime, history_with = _train_one_model(
        model=model_with_regime,
        train_loader=train_loader,
        val_loader=val_loader,
        use_regime=True,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=10,
    )

    model_no_regime, history_without = _train_one_model(
        model=model_no_regime,
        train_loader=train_loader,
        val_loader=val_loader,
        use_regime=False,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=10,
    )

    metrics_with = _evaluate_model(model_with_regime, test_loader, use_regime=True, device=device)
    metrics_without = _evaluate_model(model_no_regime, test_loader, use_regime=False, device=device)

    improvement = metrics_with["directional_accuracy"] - metrics_without["directional_accuracy"]

    print(
        f"With regime - MSE: {metrics_with['mse']:.4f}, MAE: {metrics_with['mae']:.4f}, "
        f"Direction accuracy: {metrics_with['directional_accuracy']:.1f}%"
    )
    print(
        f"Without regime - MSE: {metrics_without['mse']:.4f}, MAE: {metrics_without['mae']:.4f}, "
        f"Direction accuracy: {metrics_without['directional_accuracy']:.1f}%"
    )
    print(f"Regime conditioning improvement: {improvement:+.1f}% direction accuracy")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with_path = checkpoint_dir / f"forecaster_with_regime_{ticker}.pt"
    no_path = checkpoint_dir / f"forecaster_no_regime_{ticker}.pt"

    torch.save(
        {
            "state_dict": model_with_regime.state_dict(),
            "metrics": metrics_with,
            "history": history_with,
            "ticker": ticker,
        },
        with_path,
    )
    torch.save(
        {
            "state_dict": model_no_regime.state_dict(),
            "metrics": metrics_without,
            "history": history_without,
            "ticker": ticker,
        },
        no_path,
    )

    return {
        "ticker": ticker,
        "with_regime": metrics_with,
        "without_regime": metrics_without,
        "improvement_directional_accuracy": float(improvement),
        "checkpoints": {
            "with_regime": str(with_path),
            "without_regime": str(no_path),
        },
    }


def forecast_stock(
    ticker: str,
    predictor: object,
    checkpoint_dir: Path = Path("models/checkpoints"),
    n_history_days: int = 60,
    device: str = "cpu",
) -> dict:
    """Run a 5-step forecast for one ticker using the saved regime-conditioned model."""
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path = checkpoint_dir / f"forecaster_with_regime_{ticker}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Forecaster checkpoint not found: {ckpt_path}")

    model = StockForecaster(use_regime=True).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raw_dir = Path("data/raw")
    df = load_ticker(ticker=ticker, save_dir=raw_dir).sort_index()
    if len(df) < max(n_history_days, 50):
        raise ValueError(f"Not enough history for {ticker} to run forecast.")

    feats = _feature_frame(df)
    recent_feats = feats.tail(n_history_days).dropna()
    if len(recent_feats) < 30:
        raise ValueError("Not enough valid recent feature rows after rolling computations.")

    x_window = recent_feats.tail(30).to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(x_window[None, :, :], dtype=torch.float32, device=device)

    current_date = pd.to_datetime(df.index.max())
    regime_probs = get_regime_for_dates(np.array([np.datetime64(current_date)]), predictor=predictor)
    regime_tensor = torch.tensor(regime_probs, dtype=torch.float32, device=device)

    with torch.no_grad():
        forecast_returns = model(x_tensor, regime_tensor).cpu().numpy().reshape(-1)

    current_price = float(df["Close"].iloc[-1])

    forecast_prices: list[float] = []
    running_price = current_price
    for r in forecast_returns:
        running_price = running_price * (1.0 + float(r))
        forecast_prices.append(float(running_price))

    forecast_returns_list = [float(r) for r in forecast_returns]

    sigma = float(np.std(forecast_returns)) if len(forecast_returns) > 1 else 0.0
    total_move = float(np.sum(forecast_returns))
    lower = total_move - 1.96 * sigma
    upper = total_move + 1.96 * sigma

    # Use the last available date in our data instead of today.
    recent_predictions = predictor.predict(
        start_date="2024-10-01",
        end_date="2024-12-31",
    )
    if recent_predictions.empty:
        raise ValueError("No regime predictions available in 2024-10-01 to 2024-12-31")

    current_regime_row = recent_predictions.iloc[-1]
    current_regime_info = {
        "regime": current_regime_row["regime"],
        "regime_id": current_regime_row["regime_id"],
        "confidence": current_regime_row["confidence"],
        "p_growth": current_regime_row["p_growth"],
        "p_transition": current_regime_row["p_transition"],
        "p_panic": current_regime_row["p_panic"],
    }

    return {
        "ticker": ticker,
        "current_price": current_price,
        "current_regime": str(current_regime_info.get("regime", "Unknown")),
        "forecast_returns": forecast_returns_list,
        "forecast_prices": forecast_prices,
        "confidence_interval": (float(lower), float(upper)),
        "direction": "UP" if total_move >= 0 else "DOWN",
        "magnitude": float(total_move),
    }


if __name__ == "__main__":
    run_device = "cuda" if torch.cuda.is_available() else "cpu"

    results = train_forecaster(
        ticker="SPY",
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        device=run_device,
    )

    print("\nRegime vs no-regime model comparison")
    print(
        f"With regime direction accuracy: "
        f"{results['with_regime']['directional_accuracy']:.1f}%"
    )
    print(
        f"Without regime direction accuracy: "
        f"{results['without_regime']['directional_accuracy']:.1f}%"
    )
    print(
        f"Improvement: {results['improvement_directional_accuracy']:+.1f}%"
    )

    predictor = MarketRegimePredictor(device=run_device)
    fc = forecast_stock(ticker="SPY", predictor=predictor, device=run_device)

    print("\nSPY 5-Day Forecast")
    print(f"Current price: ${fc['current_price']:.2f}")
    print(f"Current regime: {fc['current_regime']}")
    print(f"Predicted direction: {fc['direction']}")
    print(f"Expected move: {fc['magnitude']:+.1%}")

    for i, (price, ret) in enumerate(zip(fc["forecast_prices"], fc["forecast_returns"]), start=1):
        print(f"Day {i}: ${price:.2f} ({ret:+.1%})")
