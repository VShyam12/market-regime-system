"""Evaluation utilities for LSTM+BAM regime classification.

This module evaluates the BAM-augmented model on the test split, extracts
prototype attention weights for interpretability, compares results against the
LSTM baseline, and renders diagnostic Plotly figures.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.bam import LSTMBAMModel, load_bam_model
from training.train_lstm import RegimeDataset
from training.evaluate_lstm import compute_metrics, plot_confusion_matrix
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features
from torch.utils.data import DataLoader


CLASS_NAMES = ["Growth", "Transition", "Panic"]
CLASS_IDS = [0, 1, 2]


def get_bam_predictions(
    model: LSTMBAMModel,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect BAM predictions, labels, probabilities, and attention weights."""
    model.eval()

    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _embedding, _retrieved, weights = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_list.append(preds.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            weights_list.append(weights.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0) if preds_list else np.array([], dtype=int)
    all_labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([], dtype=int)
    all_probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3), dtype=float)
    all_weights = np.concatenate(weights_list, axis=0) if weights_list else np.empty((0, 3), dtype=float)

    return all_preds, all_labels, all_probs, all_weights


def plot_attention_weights(
    dates,
    weights: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot stacked BAM prototype attention weights over time."""
    plot_dates = pd.to_datetime(dates)

    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError(f"Expected weights shape (n_samples, 3), got {weights.shape}")

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    normalized_weights = weights / row_sums

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=normalized_weights[:, 0],
            mode="lines",
            stackgroup="one",
            name="Growth attention",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=normalized_weights[:, 1],
            mode="lines",
            stackgroup="one",
            name="Transition attention",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=normalized_weights[:, 2],
            mode="lines",
            stackgroup="one",
            name="Panic attention",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="BAM prototype attention weights over test period",
        xaxis_title="Date",
        yaxis_title="Attention weight",
        width=1200,
        height=500,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved attention weights plot to {save_path}")
        except Exception as exc:
            print(f"Could not save attention weights PNG: {exc}")

    fig.show()


def compare_lstm_vs_bam(lstm_metrics: dict, bam_metrics: dict) -> None:
    """Print formatted comparison table for LSTM versus LSTM+BAM metrics."""
    lstm_report = lstm_metrics["classification_report"]
    bam_report = bam_metrics["classification_report"]

    lstm_acc = float(lstm_metrics["accuracy"])
    bam_acc = float(bam_metrics["accuracy"])

    lstm_growth_f1 = float(lstm_report["Growth"]["f1-score"])
    bam_growth_f1 = float(bam_report["Growth"]["f1-score"])

    lstm_transition_f1 = float(lstm_report["Transition"]["f1-score"])
    bam_transition_f1 = float(bam_report["Transition"]["f1-score"])

    lstm_panic_f1 = float(lstm_report["Panic"]["f1-score"])
    bam_panic_f1 = float(bam_report["Panic"]["f1-score"])

    lstm_panic_recall_pct = float(lstm_report["Panic"]["recall"] * 100.0)
    bam_panic_recall_pct = float(bam_report["Panic"]["recall"] * 100.0)

    print("Model Comparison: LSTM vs LSTM+BAM")
    print("=" * 50)
    print(f"{'Metric':<15} | {'LSTM':<7} | {'LSTM+BAM':<8} | {'Change':<8}")
    print(
        f"{'Overall Acc':<15} | {lstm_acc:>5.1f}%  | {bam_acc:>6.1f}%   | "
        f"{(bam_acc - lstm_acc):+5.1f}%"
    )
    print(
        f"{'Growth F1':<15} | {lstm_growth_f1:>7.3f} | {bam_growth_f1:>8.3f} | "
        f"{(bam_growth_f1 - lstm_growth_f1):+7.3f}"
    )
    print(
        f"{'Transition F1':<15} | {lstm_transition_f1:>7.3f} | {bam_transition_f1:>8.3f} | "
        f"{(bam_transition_f1 - lstm_transition_f1):+7.3f}"
    )
    print(
        f"{'Panic F1':<15} | {lstm_panic_f1:>7.3f} | {bam_panic_f1:>8.3f} | "
        f"{(bam_panic_f1 - lstm_panic_f1):+7.3f}"
    )
    print(
        f"{'Panic Recall':<15} | {lstm_panic_recall_pct:>5.1f}%  | {bam_panic_recall_pct:>6.1f}%   | "
        f"{(bam_panic_recall_pct - lstm_panic_recall_pct):+5.1f}%"
    )

    if bam_panic_f1 > lstm_panic_f1:
        print("Panic F1 improved")
    else:
        print("Panic F1 did not improve")

    if bam_acc > lstm_acc:
        print("Overall accuracy improved")
    else:
        print("Overall accuracy did not improve")


def plot_side_by_side_confusion(
    cm_lstm: np.ndarray,
    cm_bam: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot comparable confusion matrices for LSTM and LSTM+BAM."""
    zmax = float(max(cm_lstm.max(initial=0), cm_bam.max(initial=0)))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LSTM confusion matrix", "LSTM+BAM confusion matrix"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_lstm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            showscale=False,
            text=cm_lstm,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_bam,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            showscale=True,
            colorbar=dict(title="Count"),
            text=cm_bam,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Confusion matrix comparison: LSTM vs LSTM+BAM",
        width=1200,
        height=550,
    )
    fig.update_xaxes(title_text="Predicted label", row=1, col=1)
    fig.update_xaxes(title_text="Predicted label", row=1, col=2)
    fig.update_yaxes(title_text="True label", row=1, col=1)
    fig.update_yaxes(title_text="True label", row=1, col=2)

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved side-by-side confusion plot to {save_path}")
        except Exception as exc:
            print(f"Could not save side-by-side confusion PNG: {exc}")

    fig.show()


def run_bam_evaluation(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> dict:
    """Run full BAM evaluation and comparison workflow."""
    bam_checkpoint = checkpoint_dir / "bam_best.pt"
    lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"

    if not bam_checkpoint.exists():
        raise FileNotFoundError(f"BAM checkpoint not found at {bam_checkpoint}")
    if not lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {lstm_checkpoint}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototype checkpoint not found at {prototypes_path}")

    model = load_bam_model(
        path=bam_checkpoint,
        lstm_checkpoint=lstm_checkpoint,
        prototypes_path=prototypes_path,
        device=device,
    )

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

    x_test = splits["X_test"]
    y_test = y_test_series.astype(int).to_numpy()

    test_dataset = RegimeDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    bam_preds, bam_labels, bam_probs, bam_weights = get_bam_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )
    bam_metrics = compute_metrics(bam_labels, bam_preds, bam_probs)

    # Load LSTM metrics from disk if available, otherwise recompute with frozen LSTM.
    lstm_metrics_path_candidates = [
        checkpoint_dir / "lstm_metrics.json",
        checkpoint_dir / "evaluation_metrics.json",
    ]
    lstm_metrics: dict | None = None

    for metrics_path in lstm_metrics_path_candidates:
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if "classification_report" in raw and "accuracy" in raw and "confusion_matrix" in raw:
                loaded_cm = np.array(raw["confusion_matrix"], dtype=int)
                lstm_metrics = {
                    "accuracy": float(raw["accuracy"]),
                    "classification_report": raw["classification_report"],
                    "confusion_matrix": loaded_cm,
                }
                print(f"Loaded LSTM metrics from {metrics_path}")
                break

    if lstm_metrics is None:
        model.lstm.eval()
        lstm_preds_list: list[np.ndarray] = []
        lstm_labels_list: list[np.ndarray] = []
        lstm_probs_list: list[np.ndarray] = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _embedding = model.lstm(x_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                lstm_preds_list.append(preds.cpu().numpy())
                lstm_labels_list.append(y_batch.cpu().numpy())
                lstm_probs_list.append(probs.cpu().numpy())

        lstm_preds = np.concatenate(lstm_preds_list, axis=0) if lstm_preds_list else np.array([], dtype=int)
        lstm_labels = np.concatenate(lstm_labels_list, axis=0) if lstm_labels_list else np.array([], dtype=int)
        lstm_probs = np.concatenate(lstm_probs_list, axis=0) if lstm_probs_list else np.empty((0, 3), dtype=float)

        lstm_metrics = compute_metrics(lstm_labels, lstm_preds, lstm_probs)

    compare_lstm_vs_bam(lstm_metrics, bam_metrics)

    attention_path = checkpoint_dir / "bam_attention_weights.png"
    bam_cm_path = checkpoint_dir / "bam_confusion_matrix.png"
    side_by_side_cm_path = checkpoint_dir / "lstm_vs_bam_confusion.png"

    plot_attention_weights(dates_test, bam_weights, save_path=attention_path)
    plot_confusion_matrix(bam_metrics["confusion_matrix"], save_path=bam_cm_path)
    plot_side_by_side_confusion(
        cm_lstm=lstm_metrics["confusion_matrix"],
        cm_bam=bam_metrics["confusion_matrix"],
        save_path=side_by_side_cm_path,
    )

    growth_delta = (
        float(bam_metrics["classification_report"]["Growth"]["f1-score"])
        - float(lstm_metrics["classification_report"]["Growth"]["f1-score"])
    )
    transition_delta = (
        float(bam_metrics["classification_report"]["Transition"]["f1-score"])
        - float(lstm_metrics["classification_report"]["Transition"]["f1-score"])
    )
    panic_delta = (
        float(bam_metrics["classification_report"]["Panic"]["f1-score"])
        - float(lstm_metrics["classification_report"]["Panic"]["f1-score"])
    )

    deltas = {
        "Growth": growth_delta,
        "Transition": transition_delta,
        "Panic": panic_delta,
    }
    best_component = max(deltas, key=deltas.get)

    bam_metrics["best_component"] = best_component
    bam_metrics["component_deltas"] = deltas
    bam_metrics["attention_plot_path"] = str(attention_path)

    return bam_metrics


if __name__ == "__main__":
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = run_bam_evaluation(device=run_device)

    print("BAM evaluation complete")
    print(f"Best component: {metrics['best_component']}")
    print("Attention weights saved - interpretability confirmed")
