"""Walk-forward evaluation for the Market Regime Detection System.

This module measures temporal robustness by evaluating year-by-year performance
on the held-out test period after applying the full inference stack:
BAM probabilities -> Markov Viterbi smoothing -> VIX-based panic override.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report

from training.apply_markov import (
    get_bam_probabilities,
    hybrid_probabilities,
    rule_based_panic_override,
    viterbi_decode_with_probs,
)
from models.markov import load_markov_params
from models.bam import LSTMBAMModel
from training.train_lstm import RegimeDataset
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features
from torch.utils.data import DataLoader


CLASS_NAMES = ["Growth", "Transition", "Panic"]


def _evaluate_subset(
    probs_subset: np.ndarray,
    labels_subset: np.ndarray,
    dates_subset: np.ndarray,
    markov_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply full post-processing stack and return predictions plus metrics dict."""
    transition_matrix, _emission_matrix, initial_dist = markov_params

    hybrid_probs = hybrid_probabilities(
        probs=probs_subset,
        dates=dates_subset,
        raw_dir=raw_dir,
        vix_threshold=30.0,
        vix_panic_prob=0.6,
    )

    markov_preds, _confidence = viterbi_decode_with_probs(
        prob_sequence=hybrid_probs,
        transition_matrix=transition_matrix,
        initial_dist=initial_dist,
    )

    final_preds = rule_based_panic_override(
        smoothed_predictions=markov_preds,
        dates=dates_subset,
        raw_dir=raw_dir,
        vix_panic_threshold=30.0,
        min_consecutive_days=3,
    )

    report = classification_report(
        labels_subset,
        final_preds,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "overall_acc": float((labels_subset == final_preds).mean() * 100.0),
        "growth_f1": float(report["Growth"]["f1-score"]),
        "transition_f1": float(report["Transition"]["f1-score"]),
        "panic_f1": float(report["Panic"]["f1-score"]),
        "panic_recall": float(report["Panic"]["recall"]),
        "panic_precision": float(report["Panic"]["precision"]),
    }

    return markov_preds, final_preds, metrics


def evaluate_year(
    year: int,
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    all_dates: np.ndarray,
    markov_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_dir: Path = Path("data/raw"),
) -> dict | None:
    """Evaluate one calendar year using BAM+Markov+VIX override pipeline."""
    dt_index = pd.to_datetime(all_dates)
    mask = dt_index.year == year

    probs_year = all_probs[mask]
    labels_year = all_labels[mask]
    dates_year = dt_index[mask].to_numpy()

    if probs_year.shape[0] < 20:
        print(f"Skipping {year}: only {probs_year.shape[0]} samples")
        return None

    _markov_preds, final_preds, metrics = _evaluate_subset(
        probs_subset=probs_year,
        labels_subset=labels_year,
        dates_subset=dates_year,
        markov_params=markov_params,
        raw_dir=raw_dir,
    )

    _ = final_preds  # kept for readability and potential debugging hooks

    return {
        "year": year,
        "n_samples": int(probs_year.shape[0]),
        "overall_acc": metrics["overall_acc"],
        "growth_f1": metrics["growth_f1"],
        "transition_f1": metrics["transition_f1"],
        "panic_f1": metrics["panic_f1"],
        "panic_recall": metrics["panic_recall"],
        "panic_precision": metrics["panic_precision"],
    }


def run_walk_forward_evaluation(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> pd.DataFrame:
    """Run year-by-year and all-period walk-forward evaluation."""
    bam_checkpoint = checkpoint_dir / "bam_best.pt"
    lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"

    if not bam_checkpoint.exists():
        raise FileNotFoundError(f"BAM checkpoint not found at {bam_checkpoint}")
    if not lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {lstm_checkpoint}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototypes checkpoint not found at {prototypes_path}")

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

    all_probs, all_labels, all_dates = get_bam_probabilities(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    markov_params = load_markov_params(checkpoint_dir)

    rows: list[dict] = []
    for year in [2021, 2022, 2023, 2024]:
        year_result = evaluate_year(
            year=year,
            all_probs=all_probs,
            all_labels=all_labels,
            all_dates=all_dates,
            markov_params=markov_params,
            raw_dir=Path("data/raw"),
        )
        if year_result is not None:
            rows.append(year_result)

    _markov_all, final_all, all_metrics = _evaluate_subset(
        probs_subset=all_probs,
        labels_subset=all_labels,
        dates_subset=all_dates,
        markov_params=markov_params,
        raw_dir=Path("data/raw"),
    )

    rows.append(
        {
            "year": "All years",
            "n_samples": int(all_probs.shape[0]),
            "overall_acc": all_metrics["overall_acc"],
            "growth_f1": all_metrics["growth_f1"],
            "transition_f1": all_metrics["transition_f1"],
            "panic_f1": all_metrics["panic_f1"],
            "panic_recall": all_metrics["panic_recall"],
            "panic_precision": all_metrics["panic_precision"],
        }
    )

    results_df = pd.DataFrame(rows)
    results_df.attrs["all_dates"] = np.asarray(all_dates)
    results_df.attrs["all_labels"] = np.asarray(all_labels)
    results_df.attrs["final_predictions"] = np.asarray(final_all)

    return results_df


def print_walk_forward_table(results_df: pd.DataFrame) -> None:
    """Print formatted walk-forward table and consistency summary."""
    yearly = results_df[results_df["year"] != "All years"].copy()
    all_row = results_df[results_df["year"] == "All years"].copy()

    avg_samples = int(yearly["n_samples"].mean()) if not yearly.empty else 0
    avg_overall = float(yearly["overall_acc"].mean()) if not yearly.empty else 0.0
    avg_growth_f1 = float(yearly["growth_f1"].mean()) if not yearly.empty else 0.0
    avg_trans_f1 = float(yearly["transition_f1"].mean()) if not yearly.empty else 0.0
    avg_panic_f1 = float(yearly["panic_f1"].mean()) if not yearly.empty else 0.0
    avg_panic_rec = float(yearly["panic_recall"].mean()) if not yearly.empty else 0.0

    print("Walk-Forward Evaluation Results")
    print("=" * 70)
    print("Year  | Samples | Overall | Growth F1 | Trans F1 | Panic F1 | Panic Rec")

    for _, row in yearly.iterrows():
        print(
            f"{int(row['year']):<5} | "
            f"{int(row['n_samples']):>7} | "
            f"{row['overall_acc']:>6.1f}% | "
            f"{row['growth_f1']:>9.3f} | "
            f"{row['transition_f1']:>8.3f} | "
            f"{row['panic_f1']:>8.3f} | "
            f"{row['panic_recall'] * 100.0:>7.1f}%"
        )

    print("------+----------+---------+-----------+----------+----------+---------")
    print(
        f"{'Avg':<5} | "
        f"{avg_samples:>7} | "
        f"{avg_overall:>6.1f}% | "
        f"{avg_growth_f1:>9.3f} | "
        f"{avg_trans_f1:>8.3f} | "
        f"{avg_panic_f1:>8.3f} | "
        f"{avg_panic_rec * 100.0:>7.1f}%"
    )

    if not all_row.empty:
        allr = all_row.iloc[0]
        print(
            f"{'All':<5} | "
            f"{int(allr['n_samples']):>7} | "
            f"{allr['overall_acc']:>6.1f}% | "
            f"{allr['growth_f1']:>9.3f} | "
            f"{allr['transition_f1']:>8.3f} | "
            f"{allr['panic_f1']:>8.3f} | "
            f"{allr['panic_recall'] * 100.0:>7.1f}%"
        )

    print("=" * 70)

    if not yearly.empty:
        best_idx = yearly["overall_acc"].idxmax()
        worst_idx = yearly["overall_acc"].idxmin()

        best_row = yearly.loc[best_idx]
        worst_row = yearly.loc[worst_idx]

        metric_stds = {
            "Growth F1": float(yearly["growth_f1"].std(ddof=0)),
            "Transition F1": float(yearly["transition_f1"].std(ddof=0)),
            "Panic F1": float(yearly["panic_f1"].std(ddof=0)),
            "Panic Recall": float(yearly["panic_recall"].std(ddof=0)),
            "Overall": float(yearly["overall_acc"].std(ddof=0)),
        }
        most_consistent_metric = min(metric_stds, key=metric_stds.get)

        print(f"Best year: {int(best_row['year'])} ({best_row['overall_acc']:.1f}% overall accuracy)")
        print(f"Worst year: {int(worst_row['year'])} ({worst_row['overall_acc']:.1f}% overall accuracy)")
        print(
            f"Most consistent metric: {most_consistent_metric} "
            f"(std={metric_stds[most_consistent_metric]:.2f})"
        )


def plot_walk_forward_results(
    results_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Plot year-wise accuracy and grouped F1 scores."""
    yearly = results_df[results_df["year"] != "All years"].copy()
    if yearly.empty:
        print("No yearly rows available for plotting.")
        return

    years = yearly["year"].astype(int).astype(str).tolist()
    acc_vals = yearly["overall_acc"].to_numpy(dtype=float)

    def _acc_color(acc: float) -> str:
        if acc > 70.0:
            return "green"
        if acc >= 60.0:
            return "orange"
        return "red"

    acc_colors = [_acc_color(v) for v in acc_vals]
    avg_acc = float(np.mean(acc_vals))

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.16,
        subplot_titles=("Overall accuracy by year", "Yearly F1 scores by regime"),
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=acc_vals,
            marker_color=acc_colors,
            text=[f"{v:.1f}%" for v in acc_vals],
            textposition="outside",
            name="Overall Accuracy",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=avg_acc, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(
        go.Bar(x=years, y=yearly["growth_f1"], name="Growth F1", marker_color="green"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=years, y=yearly["transition_f1"], name="Transition F1", marker_color="orange"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=years, y=yearly["panic_f1"], name="Panic F1", marker_color="red"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Walk-forward evaluation: year-by-year performance",
        barmode="group",
        width=1200,
        height=850,
    )
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="F1 score", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved walk-forward performance plot to {save_path}")
        except Exception as exc:
            print(f"Could not save walk-forward performance PNG: {exc}")

    fig.show()


def plot_regime_calendar(
    all_dates: np.ndarray,
    true_labels: np.ndarray,
    final_predictions: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot monthly majority regime calendar with true/pred match annotations."""
    dates = pd.to_datetime(all_dates)

    calendar_df = pd.DataFrame(
        {
            "date": dates,
            "true": true_labels.astype(int),
            "pred": final_predictions.astype(int),
            "year": dates.year,
            "month": dates.month,
        }
    )

    years = list(range(2021, 2025))
    months = list(range(1, 13))

    z = np.full((len(years), len(months)), np.nan)
    text = [["" for _ in months] for _ in years]

    for yi, y in enumerate(years):
        for mi, m in enumerate(months):
            subset = calendar_df[(calendar_df["year"] == y) & (calendar_df["month"] == m)]
            if subset.empty:
                continue

            true_majority = int(subset["true"].value_counts().idxmax())
            pred_majority = int(subset["pred"].value_counts().idxmax())
            match = true_majority == pred_majority

            z[yi, mi] = pred_majority
            true_name = CLASS_NAMES[true_majority][0]
            pred_name = CLASS_NAMES[pred_majority][0]
            text[yi][mi] = f"T:{true_name} P:{pred_name} {'✓' if match else '✗'}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            y=[str(y) for y in years],
            zmin=0,
            zmax=2,
            colorscale=[
                [0.0, "green"],
                [0.3333, "green"],
                [0.3334, "orange"],
                [0.6666, "orange"],
                [0.6667, "red"],
                [1.0, "red"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate="Year %{y}, %{x}<br>%{text}<extra></extra>",
            colorbar=dict(
                title="Predicted majority",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=CLASS_NAMES,
            ),
        )
    )

    fig.update_layout(
        title="Monthly regime calendar: true vs predicted",
        width=1200,
        height=500,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved regime calendar plot to {save_path}")
        except Exception as exc:
            print(f"Could not save regime calendar PNG: {exc}")

    fig.show()


if __name__ == "__main__":
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path("models/checkpoints")

    results_df = run_walk_forward_evaluation(
        processed_dir=Path("data/processed"),
        labels_dir=Path("data/labels"),
        checkpoint_dir=checkpoint_dir,
        device=run_device,
    )

    print_walk_forward_table(results_df)

    performance_plot_path = checkpoint_dir / "walk_forward_performance.png"
    calendar_plot_path = checkpoint_dir / "walk_forward_regime_calendar.png"

    plot_walk_forward_results(results_df, save_path=performance_plot_path)

    all_dates = np.asarray(results_df.attrs.get("all_dates", np.array([])))
    all_labels = np.asarray(results_df.attrs.get("all_labels", np.array([])))
    final_preds = np.asarray(results_df.attrs.get("final_predictions", np.array([])))
    if all_dates.size > 0 and all_labels.size > 0 and final_preds.size > 0:
        plot_regime_calendar(
            all_dates=all_dates,
            true_labels=all_labels,
            final_predictions=final_preds,
            save_path=calendar_plot_path,
        )

    csv_path = checkpoint_dir / "walk_forward_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved walk-forward results to {csv_path}")

    yearly = results_df[results_df["year"] != "All years"]
    avg_acc = float(yearly["overall_acc"].mean()) if not yearly.empty else 0.0

    panic_map = {str(int(row["year"])): float(row["panic_recall"] * 100.0) for _, row in yearly.iterrows()}

    print("Walk-forward evaluation complete")
    print(f"Average accuracy across years: {avg_acc:.1f}%")
    print(
        "Panic recall by year: "
        f"2021={panic_map.get('2021', 0.0):.1f}%, "
        f"2022={panic_map.get('2022', 0.0):.1f}%, "
        f"2023={panic_map.get('2023', 0.0):.1f}%, "
        f"2024={panic_map.get('2024', 0.0):.1f}%"
    )
