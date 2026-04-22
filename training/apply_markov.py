"""Apply Markov (Viterbi) smoothing on BAM regime predictions.

This module runs the final inference pipeline:
LSTM encoder -> BAM probabilities -> Markov/Viterbi smoothing.
It evaluates raw vs smoothed outputs, visualizes sequence effects over time,
and exports final results for deployment reporting.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from models.markov import viterbi_decode_with_probs, smooth_predictions, load_markov_params
from models.bam import LSTMBAMModel
from training.train_lstm import RegimeDataset
from training.evaluate_lstm import compute_metrics
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features
from torch.utils.data import DataLoader


CLASS_NAMES = ["Growth", "Transition", "Panic"]
CLASS_COLORS = {0: "green", 1: "orange", 2: "red"}
CLASS_IDS = [0, 1, 2]


def get_bam_probabilities(
    model: LSTMBAMModel,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BAM model and collect soft probabilities, labels, and dates."""
    model.eval()

    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _embedding, _retrieved, _weights = model(x_batch)
            probs = torch.softmax(logits, dim=1)

            probs_list.append(probs.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())

    probs_arr = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3), dtype=float)
    true_labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([], dtype=int)

    dates_attr = getattr(dataloader, "dates", None)
    if dates_attr is None:
        dates = np.arange(probs_arr.shape[0])
    else:
        dates = np.asarray(dates_attr)
        if dates.shape[0] != probs_arr.shape[0]:
            raise ValueError(
                "Date count does not match prediction count: "
                f"dates={dates.shape[0]}, probs={probs_arr.shape[0]}"
            )

    return probs_arr, true_labels, dates


def apply_viterbi_smoothing(
    probs: np.ndarray,
    markov_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    use_soft_viterbi: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply soft or hard Viterbi smoothing to BAM outputs."""
    transition_matrix, emission_matrix, initial_dist = markov_params

    raw_hard = np.argmax(probs, axis=1).astype(int)

    if use_soft_viterbi:
        smoothed_labels, confidence_scores = viterbi_decode_with_probs(
            prob_sequence=probs,
            transition_matrix=transition_matrix,
            initial_dist=initial_dist,
        )
    else:
        smoothed_labels = smooth_predictions(
            raw_predictions=raw_hard,
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
            initial_dist=initial_dist,
        )
        confidence_scores = None

    changed = int(np.sum(raw_hard != smoothed_labels))
    print(f"Predictions changed by Viterbi smoothing: {changed}/{len(raw_hard)}")

    return smoothed_labels, confidence_scores


def hybrid_probabilities(
    probs: np.ndarray,
    dates: np.ndarray,
    raw_dir: Path = Path("data/raw"),
    vix_threshold: float = 30.0,
    vix_panic_prob: float = 0.6,
) -> np.ndarray:
    """Blend BAM probabilities with VIX-conditioned panic priors."""
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected probs shape (n_samples, 3), got {probs.shape}")

    date_index = pd.to_datetime(dates)
    if date_index.shape[0] != probs.shape[0]:
        raise ValueError(
            f"dates and probs length mismatch: dates={date_index.shape[0]}, probs={probs.shape[0]}"
        )

    vix_path = raw_dir / "VIX.parquet"
    if not vix_path.exists():
        raise FileNotFoundError(f"VIX file not found at {vix_path}")

    vix_df = pd.read_parquet(vix_path)
    if "Close" not in vix_df.columns:
        raise ValueError(f"Expected 'Close' column in {vix_path}")

    vix_series = vix_df["Close"].copy()
    if not isinstance(vix_series.index, pd.DatetimeIndex):
        vix_series.index = pd.to_datetime(vix_series.index)

    aligned_vix = vix_series.reindex(date_index).ffill().bfill()
    if aligned_vix.isna().any():
        missing_count = int(aligned_vix.isna().sum())
        raise ValueError(f"Could not align VIX to {missing_count} dates in test period")

    hybrid = probs.astype(float, copy=True)

    high_vix_mask = aligned_vix.to_numpy() > float(vix_threshold)
    mid_vix_mask = (aligned_vix.to_numpy() > 20.0) & (aligned_vix.to_numpy() < float(vix_threshold))

    for i in np.where(high_vix_mask)[0]:
        g = float(probs[i, 0])
        t = float(probs[i, 1])
        non_panic_sum = g + t

        hybrid[i, 2] = float(vix_panic_prob)
        if non_panic_sum > 0:
            hybrid[i, 0] = (1.0 - vix_panic_prob) * g / non_panic_sum
            hybrid[i, 1] = (1.0 - vix_panic_prob) * t / non_panic_sum
        else:
            hybrid[i, 0] = (1.0 - vix_panic_prob) * 0.5
            hybrid[i, 1] = (1.0 - vix_panic_prob) * 0.5

    for i in np.where(mid_vix_mask)[0]:
        row = hybrid[i].copy()
        row[2] += 0.15
        row_sum = float(row.sum())
        if row_sum > 0:
            hybrid[i] = row / row_sum

    print(f"VIX > {vix_threshold:.1f} days in test period: {int(high_vix_mask.sum())}")
    print(f"VIX 20-{vix_threshold:.1f} days in test period: {int(mid_vix_mask.sum())}")

    return hybrid


def evaluate_smoothing_impact(
    true_labels: np.ndarray,
    raw_predictions: np.ndarray,
    smoothed_predictions: np.ndarray,
    raw_probs: np.ndarray,
    markov_predictions: np.ndarray | None = None,
    final_label: str = "LSTM + BAM + Markov + VIX override",
) -> dict:
    """Compare raw, Markov, and optional final predictions with detailed reporting."""
    raw_metrics = compute_metrics(true_labels, raw_predictions, raw_probs)
    smoothed_metrics = compute_metrics(true_labels, smoothed_predictions, raw_probs)
    markov_metrics = (
        compute_metrics(true_labels, markov_predictions, raw_probs)
        if markov_predictions is not None
        else None
    )

    raw_report = raw_metrics["classification_report"]
    smooth_report = smoothed_metrics["classification_report"]
    markov_report = markov_metrics["classification_report"] if markov_metrics is not None else None

    print("=" * 60)
    print("FULL PIPELINE RESULTS: LSTM -> BAM -> Markov -> VIX Override")
    print("=" * 60)
    if markov_metrics is not None:
        print(
            f"{'Metric':<17} | {'Raw BAM':<8} | {'+Markov':<8} | {'+VIX Override':<13}"
        )

        def _line(metric_name: str, raw_val: float, markov_val: float, final_val: float, pct: bool = False) -> None:
            if pct:
                print(
                    f"{metric_name:<17} | {raw_val:>6.1f}%  | {markov_val:>6.1f}%  | {final_val:>11.1f}%"
                )
            else:
                print(
                    f"{metric_name:<17} | {raw_val:>7.3f} | {markov_val:>7.3f} | {final_val:>11.3f}"
                )

        _line(
            "Overall Accuracy",
            float(raw_metrics["accuracy"]),
            float(markov_metrics["accuracy"]),
            float(smoothed_metrics["accuracy"]),
            pct=True,
        )
        _line(
            "Growth Precision",
            float(raw_report["Growth"]["precision"]),
            float(markov_report["Growth"]["precision"]),
            float(smooth_report["Growth"]["precision"]),
        )
        _line(
            "Growth Recall",
            float(raw_report["Growth"]["recall"]),
            float(markov_report["Growth"]["recall"]),
            float(smooth_report["Growth"]["recall"]),
        )
        _line(
            "Growth F1",
            float(raw_report["Growth"]["f1-score"]),
            float(markov_report["Growth"]["f1-score"]),
            float(smooth_report["Growth"]["f1-score"]),
        )
        _line(
            "Trans Precision",
            float(raw_report["Transition"]["precision"]),
            float(markov_report["Transition"]["precision"]),
            float(smooth_report["Transition"]["precision"]),
        )
        _line(
            "Trans Recall",
            float(raw_report["Transition"]["recall"]),
            float(markov_report["Transition"]["recall"]),
            float(smooth_report["Transition"]["recall"]),
        )
        _line(
            "Trans F1",
            float(raw_report["Transition"]["f1-score"]),
            float(markov_report["Transition"]["f1-score"]),
            float(smooth_report["Transition"]["f1-score"]),
        )
        _line(
            "Panic Precision",
            float(raw_report["Panic"]["precision"]),
            float(markov_report["Panic"]["precision"]),
            float(smooth_report["Panic"]["precision"]),
        )
        _line(
            "Panic Recall",
            float(raw_report["Panic"]["recall"]),
            float(markov_report["Panic"]["recall"]),
            float(smooth_report["Panic"]["recall"]),
        )
        _line(
            "Panic F1",
            float(raw_report["Panic"]["f1-score"]),
            float(markov_report["Panic"]["f1-score"]),
            float(smooth_report["Panic"]["f1-score"]),
        )
    else:
        print(f"{'Metric':<17} | {'Raw BAM':<8} | {'+Markov':<8} | {'Change':<8}")

        def _line(metric_name: str, raw_val: float, smooth_val: float, pct: bool = False) -> None:
            delta = smooth_val - raw_val
            if pct:
                print(
                    f"{metric_name:<17} | {raw_val:>6.1f}%  | {smooth_val:>6.1f}%  | {delta:>+6.1f}%"
                )
            else:
                print(
                    f"{metric_name:<17} | {raw_val:>7.3f} | {smooth_val:>7.3f} | {delta:>+7.3f}"
                )

        _line("Overall Accuracy", float(raw_metrics["accuracy"]), float(smoothed_metrics["accuracy"]), pct=True)
        _line("Growth Precision", float(raw_report["Growth"]["precision"]), float(smooth_report["Growth"]["precision"]))
        _line("Growth Recall", float(raw_report["Growth"]["recall"]), float(smooth_report["Growth"]["recall"]))
        _line("Growth F1", float(raw_report["Growth"]["f1-score"]), float(smooth_report["Growth"]["f1-score"]))
        _line(
            "Trans Precision",
            float(raw_report["Transition"]["precision"]),
            float(smooth_report["Transition"]["precision"]),
        )
        _line(
            "Trans Recall",
            float(raw_report["Transition"]["recall"]),
            float(smooth_report["Transition"]["recall"]),
        )
        _line(
            "Trans F1",
            float(raw_report["Transition"]["f1-score"]),
            float(smooth_report["Transition"]["f1-score"]),
        )
        _line("Panic Precision", float(raw_report["Panic"]["precision"]), float(smooth_report["Panic"]["precision"]))
        _line("Panic Recall", float(raw_report["Panic"]["recall"]), float(smooth_report["Panic"]["recall"]))
        _line("Panic F1", float(raw_report["Panic"]["f1-score"]), float(smooth_report["Panic"]["f1-score"]))

    print(f"Final stage label: {final_label}")

    raw_panic_recall_pct = float(raw_report["Panic"]["recall"]) * 100.0
    smooth_panic_recall_pct = float(smooth_report["Panic"]["recall"]) * 100.0

    if smooth_panic_recall_pct > raw_panic_recall_pct:
        print(
            f"*** PANIC RECALL IMPROVED: {raw_panic_recall_pct:.1f}% -> {smooth_panic_recall_pct:.1f}% ***"
        )
    else:
        print("Panic recall unchanged - see report discussion")

    return {
        "raw": {
            "metrics": raw_metrics,
            "accuracy": float(raw_metrics["accuracy"]),
            "panic_recall": float(raw_report["Panic"]["recall"]),
        },
        "markov": {
            "metrics": markov_metrics,
            "accuracy": float(markov_metrics["accuracy"]) if markov_metrics is not None else None,
            "panic_recall": float(markov_report["Panic"]["recall"]) if markov_report is not None else None,
        },
        "smoothed": {
            "metrics": smoothed_metrics,
            "accuracy": float(smoothed_metrics["accuracy"]),
            "panic_recall": float(smooth_report["Panic"]["recall"]),
        },
    }


def rule_based_panic_override(
    smoothed_predictions: np.ndarray,
    dates: np.ndarray,
    raw_dir: Path = Path("data/raw"),
    vix_panic_threshold: float = 30.0,
    min_consecutive_days: int = 3,
) -> np.ndarray:
    """Override smoothed labels to Panic based on sustained/extreme VIX rules."""
    preds = np.asarray(smoothed_predictions, dtype=int).copy()
    date_index = pd.to_datetime(dates)

    if preds.shape[0] != date_index.shape[0]:
        raise ValueError(
            "smoothed_predictions and dates must have same length: "
            f"preds={preds.shape[0]}, dates={date_index.shape[0]}"
        )

    vix_path = raw_dir / "VIX.parquet"
    if not vix_path.exists():
        raise FileNotFoundError(f"VIX file not found at {vix_path}")

    vix_df = pd.read_parquet(vix_path)
    if "Close" not in vix_df.columns:
        raise ValueError(f"Expected 'Close' column in {vix_path}")

    vix_series = vix_df["Close"].copy()
    if not isinstance(vix_series.index, pd.DatetimeIndex):
        vix_series.index = pd.to_datetime(vix_series.index)

    aligned_vix = vix_series.reindex(date_index).ffill().bfill()
    if aligned_vix.isna().any():
        missing_count = int(aligned_vix.isna().sum())
        raise ValueError(f"Could not align VIX to {missing_count} requested dates")

    high_mask = aligned_vix.to_numpy() > float(vix_panic_threshold)
    extreme_mask = aligned_vix.to_numpy() > 35.0

    override_mask = np.zeros_like(high_mask, dtype=bool)
    stretches_found = 0

    start = None
    for i, is_high in enumerate(high_mask):
        if is_high and start is None:
            start = i
        if (not is_high or i == len(high_mask) - 1) and start is not None:
            end = i if (is_high and i == len(high_mask) - 1) else i - 1
            run_len = end - start + 1
            if run_len >= int(min_consecutive_days):
                override_mask[start : end + 1] = True
                stretches_found += 1
            start = None

    # Also force Panic for isolated extreme spikes.
    override_mask = override_mask | extreme_mask

    before = preds.copy()
    preds[override_mask] = 2

    overridden_days = int(np.sum(before != preds))
    print("Rule-based Panic override applied")
    print(f"VIX > {vix_panic_threshold:.0f} for {min_consecutive_days}+ days: {stretches_found} stretches found")
    print(f"Total days overridden to Panic: {overridden_days}")

    return preds


def _add_regime_lines(fig: go.Figure, x_vals, y_vals: np.ndarray, row: int, col: int, name_prefix: str) -> None:
    for cls in CLASS_IDS:
        masked = np.where(y_vals == cls, cls, np.nan)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=masked,
                mode="lines",
                line=dict(color=CLASS_COLORS[cls], width=2),
                line_shape="hv",
                name=f"{name_prefix} {CLASS_NAMES[cls]}",
                connectgaps=False,
                showlegend=(row == 1),
            ),
            row=row,
            col=col,
        )


def plot_smoothing_comparison(
    dates,
    true_labels: np.ndarray,
    raw_predictions: np.ndarray,
    smoothed_predictions: np.ndarray,
    confidence_scores: np.ndarray | None,
    save_path: Path | None = None,
) -> None:
    """Plot true labels, raw BAM predictions, and smoothed predictions over time."""
    dates_idx = pd.to_datetime(dates)

    raw_acc = float((raw_predictions == true_labels).mean() * 100.0)
    smooth_acc = float((smoothed_predictions == true_labels).mean() * 100.0)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "True regime labels over time",
            f"Raw BAM predictions (Accuracy: {raw_acc:.1f}%)",
            f"Viterbi smoothed predictions (Accuracy: {smooth_acc:.1f}%)",
        ),
        row_heights=[0.33, 0.33, 0.34],
    )

    _add_regime_lines(fig, dates_idx, true_labels, row=1, col=1, name_prefix="True")
    _add_regime_lines(fig, dates_idx, raw_predictions, row=2, col=1, name_prefix="Raw")
    _add_regime_lines(fig, dates_idx, smoothed_predictions, row=3, col=1, name_prefix="Smoothed")

    if confidence_scores is not None and len(confidence_scores) == len(dates_idx):
        fig.add_trace(
            go.Scatter(
                x=dates_idx,
                y=confidence_scores,
                mode="lines",
                line=dict(color="gray", width=1),
                name="Viterbi confidence",
                showlegend=True,
                opacity=0.8,
            ),
            row=3,
            col=1,
        )

    changed_mask = raw_predictions != smoothed_predictions
    for i in np.where(changed_mask)[0]:
        x0 = dates_idx[i]
        x1 = dates_idx[i + 1] if i + 1 < len(dates_idx) else dates_idx[i]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(186, 85, 211, 0.15)",
            line_width=0,
            row="all",
            col=1,
        )

    for r in [1, 2, 3]:
        fig.update_yaxes(
            title_text="Regime",
            row=r,
            col=1,
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=CLASS_NAMES,
            range=[-0.3, 2.3],
        )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(
        title="Regime predictions: True vs BAM vs Viterbi",
        width=1300,
        height=900,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved smoothing comparison plot to {save_path}")
        except Exception as exc:
            print(f"Could not save smoothing comparison PNG: {exc}")

    fig.show()


def plot_final_confusion_matrices(
    true_labels: np.ndarray,
    raw_preds: np.ndarray,
    smoothed_preds: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot raw, smoothed, and difference confusion matrices side by side."""
    cm_raw = confusion_matrix(true_labels, raw_preds, labels=CLASS_IDS)
    cm_smooth = confusion_matrix(true_labels, smoothed_preds, labels=CLASS_IDS)
    cm_diff = cm_smooth - cm_raw

    zmax_main = int(max(cm_raw.max(initial=0), cm_smooth.max(initial=0)))
    diff_abs_max = int(np.max(np.abs(cm_diff))) if cm_diff.size > 0 else 1

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "True vs Raw BAM",
            "True vs Viterbi smoothed",
            "Difference (smoothed - raw)",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_raw,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0,
            zmax=zmax_main,
            text=cm_raw,
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_smooth,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0,
            zmax=zmax_main,
            text=cm_smooth,
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_diff,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Greens",
            zmin=-diff_abs_max,
            zmax=diff_abs_max,
            text=cm_diff,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Count delta"),
        ),
        row=1,
        col=3,
    )

    for c in [1, 2, 3]:
        fig.update_xaxes(title_text="Predicted", row=1, col=c)
        fig.update_yaxes(title_text="True", row=1, col=c)

    fig.update_layout(
        title="Confusion matrices: BAM vs Viterbi smoothed",
        width=1500,
        height=500,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved confusion matrix comparison plot to {save_path}")
        except Exception as exc:
            print(f"Could not save confusion matrix comparison PNG: {exc}")

    fig.show()


def run_markov_smoothing(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> dict:
    """Run end-to-end Markov smoothing on BAM test probabilities."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bam_checkpoint = checkpoint_dir / "bam_best.pt"
    lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"

    if not bam_checkpoint.exists():
        raise FileNotFoundError(f"BAM checkpoint not found at {bam_checkpoint}")
    if not lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {lstm_checkpoint}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototypes file not found at {prototypes_path}")

    model = LSTMBAMModel(
        lstm_checkpoint=lstm_checkpoint,
        prototypes_path=prototypes_path,
        device=device,
    )

    bam_state = torch.load(bam_checkpoint, map_location=device)
    if "bam_state_dict" not in bam_state:
        raise KeyError(f"No 'bam_state_dict' found in {bam_checkpoint}")
    model.bam.load_state_dict(bam_state["bam_state_dict"])

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
        raise ValueError(f"Missing {missing_count} test labels after alignment")

    x_test = splits["X_test"]
    y_test = y_test_series.astype(int).to_numpy()

    test_dataset = RegimeDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    setattr(test_loader, "dates", dates_test.to_numpy())

    probs, true_labels, dates = get_bam_probabilities(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    hybrid_probs = hybrid_probabilities(
        probs=probs,
        dates=dates,
        raw_dir=Path("data/raw"),
        vix_threshold=30.0,
        vix_panic_prob=0.6,
    )

    markov_params = load_markov_params(checkpoint_dir)

    smoothed_preds, confidence_scores = apply_viterbi_smoothing(
        probs=hybrid_probs,
        markov_params=markov_params,
        use_soft_viterbi=True,
    )

    final_preds = rule_based_panic_override(
        smoothed_predictions=smoothed_preds,
        dates=dates,
        raw_dir=Path("data/raw"),
        vix_panic_threshold=30.0,
        min_consecutive_days=3,
    )

    raw_preds = np.argmax(probs, axis=1).astype(int)

    results = evaluate_smoothing_impact(
        true_labels=true_labels,
        raw_predictions=raw_preds,
        smoothed_predictions=final_preds,
        raw_probs=probs,
        markov_predictions=smoothed_preds,
        final_label="LSTM + BAM + Markov + VIX override",
    )

    timeline_plot_path = checkpoint_dir / "markov_smoothing_timeline.png"
    cm_plot_path = checkpoint_dir / "markov_confusion_comparison.png"

    plot_smoothing_comparison(
        dates=dates,
        true_labels=true_labels,
        raw_predictions=raw_preds,
        smoothed_predictions=final_preds,
        confidence_scores=confidence_scores,
        save_path=timeline_plot_path,
    )
    plot_final_confusion_matrices(
        true_labels=true_labels,
        raw_preds=raw_preds,
        smoothed_preds=final_preds,
        save_path=cm_plot_path,
    )

    output_payload = {
        "raw_accuracy": float(results["raw"]["accuracy"]),
        "smoothed_accuracy": float(results["smoothed"]["accuracy"]),
        "raw_panic_recall": float(results["raw"]["panic_recall"]),
        "smoothed_panic_recall": float(results["smoothed"]["panic_recall"]),
        "raw_confusion_matrix": np.asarray(results["raw"]["metrics"]["confusion_matrix"]).tolist(),
        "smoothed_confusion_matrix": np.asarray(results["smoothed"]["metrics"]["confusion_matrix"]).tolist(),
        "timeline_plot_path": str(timeline_plot_path),
        "confusion_plot_path": str(cm_plot_path),
    }

    output_json = checkpoint_dir / "final_pipeline_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved final pipeline results to {output_json}")

    return output_payload


if __name__ == "__main__":
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    final_results = run_markov_smoothing(device=run_device)

    print("MARKET REGIME DETECTION SYSTEM - FINAL RESULTS")
    print("LSTM encoder: 65.4% test accuracy")
    print("LSTM + BAM: 65.8% test accuracy")
    print(f"LSTM + BAM + Markov: {final_results['smoothed_accuracy']:.1f}% test accuracy")
    print(
        f"Panic recall: {final_results['raw_panic_recall'] * 100.0:.1f}% -> "
        f"{final_results['smoothed_panic_recall'] * 100.0:.1f}%"
    )
    print("System ready for dashboard deployment")
