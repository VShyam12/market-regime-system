"""Evaluation utilities for the trained LSTM regime classifier.

This module loads the best checkpoint, evaluates test-set predictions, computes
classification metrics, and generates diagnostic plots for confusion matrix,
training history, and regime probability dynamics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from training.train_lstm import RegimeDataset
from models.lstm import load_model
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features
from torch.utils.data import DataLoader


CLASS_NAMES = ["Growth", "Transition", "Panic"]
CLASS_IDS = [0, 1, 2]


def get_predictions(model, dataloader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference and collect predictions, labels, and probabilities."""
    model.eval()

    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_list.append(preds.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0) if preds_list else np.array([], dtype=int)
    all_labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([], dtype=int)
    all_probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3), dtype=float)

    return all_preds, all_labels, all_probs


def compute_metrics(y_true, y_pred, y_probs) -> dict:
    """Compute accuracy, classification report, and confusion matrix."""
    _ = y_probs  # Reserved for future probability-based metrics.

    accuracy = float((y_true == y_pred).mean() * 100.0) if len(y_true) > 0 else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Path | None = None) -> None:
    """Plot confusion matrix heatmap with counts and percentages."""
    total = cm.sum()
    pct = (cm / total * 100.0) if total > 0 else np.zeros_like(cm, dtype=float)

    annotations = np.array(
        [[f"{cm[i, j]}<br>{pct[i, j]:.1f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            showscale=True,
            text=annotations,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="LSTM confusion matrix — test set",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        width=800,
        height=650,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved confusion matrix plot to {save_path}")
        except Exception as exc:
            print(f"Could not save confusion matrix PNG: {exc}")

    fig.show()


def plot_training_history(history_path: Path, save_path: Path | None = None) -> None:
    """Plot training/validation loss and accuracy across epochs."""
    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found at {history_path}")

    history = pd.read_json(history_path)

    # Handle JSON dict-style history robustly.
    if history.shape[0] == 1 and "train_loss" in history.columns:
        row = history.iloc[0]
        train_loss = row["train_loss"]
        val_loss = row["val_loss"]
        train_acc = row["train_acc"]
        val_acc = row["val_acc"]
    else:
        # Fallback if JSON was loaded into expanded rows.
        with history_path.open("r", encoding="utf-8") as f:
            import json

            hist_dict = json.load(f)
        train_loss = hist_dict["train_loss"]
        val_loss = hist_dict["val_loss"]
        train_acc = hist_dict["train_acc"]
        val_acc = hist_dict["val_acc"]

    epochs = list(range(1, len(train_loss) + 1))
    best_epoch = int(np.argmin(np.array(val_loss)) + 1) if len(val_loss) > 0 else 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Loss", "Accuracy"),
    )

    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines", name="Train Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines", name="Val Loss"), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines", name="Train Acc"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines", name="Val Acc"), row=2, col=1)

    fig.add_vline(x=best_epoch, line_dash="dash", line_color="black")

    fig.update_layout(
        title="LSTM training history",
        width=1000,
        height=750,
    )
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved training history plot to {save_path}")
        except Exception as exc:
            print(f"Could not save training history PNG: {exc}")

    fig.show()


def plot_regime_probabilities(
    dates,
    probs: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot stacked class probabilities across the test period."""
    dates = pd.to_datetime(dates)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probs[:, 0],
            mode="lines",
            stackgroup="one",
            name="P(Growth)",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probs[:, 1],
            mode="lines",
            stackgroup="one",
            name="P(Transition)",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probs[:, 2],
            mode="lines",
            stackgroup="one",
            name="P(Panic)",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="LSTM regime probabilities over test period",
        xaxis_title="Date",
        yaxis_title="Probability",
        width=1200,
        height=500,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved regime probabilities plot to {save_path}")
        except Exception as exc:
            print(f"Could not save regime probabilities PNG: {exc}")

    fig.show()


def run_evaluation(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> dict:
    """Run full LSTM evaluation workflow and return computed metrics."""
    checkpoint_path = checkpoint_dir / "lstm_best.pt"
    history_path = checkpoint_dir / "training_history.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")

    model = load_model(checkpoint_path, device=device)

    splits = load_splits(processed_dir)
    labels = load_labels(labels_dir)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    features_df = pd.read_parquet(features_path)
    _, aligned_labels = align_labels_with_features(labels, features_df)

    dates_test = pd.to_datetime(splits["dates_test"])
    y_test_series = aligned_labels.reindex(dates_test)
    if y_test_series.isna().any():
        missing_count = int(y_test_series.isna().sum())
        raise ValueError(f"Missing {missing_count} test labels after alignment.")

    X_test = splits["X_test"]
    y_test = y_test_series.astype(int).to_numpy()

    test_dataset = RegimeDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_probs)

    cm_path = checkpoint_dir / "confusion_matrix.png"
    history_plot_path = checkpoint_dir / "training_history.png"
    probs_path = checkpoint_dir / "regime_probabilities.png"

    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)
    plot_training_history(history_path, save_path=history_plot_path)
    plot_regime_probabilities(dates_test, y_probs, save_path=probs_path)

    return metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = run_evaluation(device=device)

    accuracy = metrics["accuracy"]
    report = metrics["classification_report"]

    print(f"Overall accuracy: {accuracy:.2f}%")
    print(
        "Per class F1 scores: "
        f"Growth={report['Growth']['f1-score']:.3f}, "
        f"Transition={report['Transition']['f1-score']:.3f}, "
        f"Panic={report['Panic']['f1-score']:.3f}"
    )
    print(f"Panic recall: {report['Panic']['recall']:.3f}")
    print("Evaluation complete - plots saved to models/checkpoints/")
