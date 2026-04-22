"""Visual diagnostics for BAM-enhanced regime embeddings.

This module extracts pre-BAM (raw LSTM) and post-BAM (retrieved) embeddings,
projects them with t-SNE, visualizes regime clustering changes, inspects BAM
attention behavior, and reports prototype geometry.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.bam import LSTMBAMModel
from training.train_lstm import RegimeDataset
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features
from torch.utils.data import DataLoader


CLASS_NAMES = ["Growth", "Transition", "Panic"]
CLASS_COLORS = {0: "green", 1: "orange", 2: "red"}


def extract_bam_embeddings(
    model: LSTMBAMModel,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw/retrieved embeddings, attention weights, and labels."""
    model.eval()

    raw_list: list[np.ndarray] = []
    retrieved_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            _logits, embedding, retrieved, weights = model(x_batch)

            raw_list.append(embedding.cpu().numpy())
            retrieved_list.append(retrieved.cpu().numpy())
            weights_list.append(weights.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())

    raw_embeddings = np.concatenate(raw_list, axis=0) if raw_list else np.empty((0, 64), dtype=float)
    retrieved_embeddings = (
        np.concatenate(retrieved_list, axis=0) if retrieved_list else np.empty((0, 64), dtype=float)
    )
    attention_weights = np.concatenate(weights_list, axis=0) if weights_list else np.empty((0, 3), dtype=float)
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([], dtype=int)

    return raw_embeddings, retrieved_embeddings, attention_weights, labels


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D with t-SNE."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")

    effective_perplexity = min(perplexity, max(2, embeddings.shape[0] - 1))

    with tqdm(total=1, desc="Running t-SNE", unit="fit") as pbar:
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=random_state,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        result = tsne.fit_transform(embeddings)
        pbar.update(1)

    return result


def plot_tsne_comparison(
    tsne_raw: np.ndarray,
    tsne_retrieved: np.ndarray,
    labels: np.ndarray,
    prototypes_raw: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot side-by-side t-SNE views before and after BAM retrieval."""
    if len(labels) != len(tsne_raw) or len(labels) != len(tsne_retrieved):
        raise ValueError("labels, tsne_raw, and tsne_retrieved must have same number of rows.")

    if prototypes_raw.shape == (3, 4):
        prototypes_left = prototypes_raw[:, :2]
        prototypes_right = prototypes_raw[:, 2:]
    elif prototypes_raw.shape == (3, 2):
        prototypes_left = prototypes_raw
        prototypes_right = prototypes_raw
    else:
        raise ValueError(
            "Expected prototypes_raw to have shape (3, 2) or (3, 4) for prototype star positions."
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Before BAM (LSTM embeddings)", "After BAM (retrieved embeddings)"),
        horizontal_spacing=0.08,
    )

    for regime in [0, 1, 2]:
        mask = labels == regime
        fig.add_trace(
            go.Scatter(
                x=tsne_raw[mask, 0],
                y=tsne_raw[mask, 1],
                mode="markers",
                name=CLASS_NAMES[regime],
                marker=dict(color=CLASS_COLORS[regime], size=6, opacity=0.6),
                legendgroup=CLASS_NAMES[regime],
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tsne_retrieved[mask, 0],
                y=tsne_retrieved[mask, 1],
                mode="markers",
                name=CLASS_NAMES[regime],
                marker=dict(color=CLASS_COLORS[regime], size=6, opacity=0.6),
                legendgroup=CLASS_NAMES[regime],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    for regime in [0, 1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[prototypes_left[regime, 0]],
                y=[prototypes_left[regime, 1]],
                mode="markers",
                name=f"{CLASS_NAMES[regime]} prototype",
                marker=dict(
                    symbol="star",
                    color=CLASS_COLORS[regime],
                    size=20,
                    line=dict(color="black", width=2),
                ),
                legendgroup=f"{CLASS_NAMES[regime]}_prototype",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[prototypes_right[regime, 0]],
                y=[prototypes_right[regime, 1]],
                mode="markers",
                name=f"{CLASS_NAMES[regime]} prototype",
                marker=dict(
                    symbol="star",
                    color=CLASS_COLORS[regime],
                    size=20,
                    line=dict(color="black", width=2),
                ),
                legendgroup=f"{CLASS_NAMES[regime]}_prototype",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Embedding space: Before vs After BAM (t-SNE)",
        width=1300,
        height=600,
    )
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved t-SNE comparison plot to {save_path}")
        except Exception as exc:
            print(f"Could not save t-SNE comparison PNG: {exc}")

    fig.show()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot mean BAM attention matrix grouped by true regime."""
    if attention_weights.shape[0] != labels.shape[0]:
        raise ValueError("attention_weights and labels must have same number of rows")
    if attention_weights.shape[1] != 3:
        raise ValueError(f"Expected attention_weights shape (n_samples, 3), got {attention_weights.shape}")

    attention_matrix = np.zeros((3, 3), dtype=float)
    for regime in [0, 1, 2]:
        mask = labels == regime
        if np.any(mask):
            attention_matrix[regime] = attention_weights[mask].mean(axis=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=attention_matrix,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=np.round(attention_matrix, 3),
            texttemplate="%{text}",
            colorbar=dict(title="Mean attention"),
        )
    )
    fig.update_layout(
        title="Mean BAM attention by true regime",
        xaxis_title="Prototype attended to",
        yaxis_title="True regime",
        width=850,
        height=600,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved attention heatmap to {save_path}")
        except Exception as exc:
            print(f"Could not save attention heatmap PNG: {exc}")

    fig.show()

    print("\nMean BAM attention by true regime")
    print(" " * 16 + "Growth   Transition   Panic")
    for i, name in enumerate(CLASS_NAMES):
        row = attention_matrix[i]
        print(f"{name:<12} | {row[0]:>6.3f}     {row[1]:>6.3f}     {row[2]:>6.3f}")


def plot_prototype_distances(
    prototypes: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot and print pairwise L2 distances between prototypes."""
    if prototypes.shape != (3, prototypes.shape[1]):
        if prototypes.ndim != 2 or prototypes.shape[0] != 3:
            raise ValueError(f"Expected prototypes shape (3, d), got {prototypes.shape}")

    diff = prototypes[:, None, :] - prototypes[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=-1)

    fig = go.Figure(
        data=go.Heatmap(
            z=distance_matrix,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Viridis",
            text=np.round(distance_matrix, 2),
            texttemplate="%{text}",
            colorbar=dict(title="L2 distance"),
        )
    )
    fig.update_layout(
        title="Prototype pairwise distances",
        width=850,
        height=600,
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved prototype distance heatmap to {save_path}")
        except Exception as exc:
            print(f"Could not save prototype distance PNG: {exc}")

    fig.show()

    print(f"Growth ↔ Transition: {distance_matrix[0, 1]:.2f}")
    print(f"Growth ↔ Panic: {distance_matrix[0, 2]:.2f}")
    print(f"Transition ↔ Panic: {distance_matrix[1, 2]:.2f}")


def run_visualisation(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> None:
    """Run BAM embedding and attention visual diagnostics."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"
    bam_checkpoint = checkpoint_dir / "bam_best.pt"

    if not lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {lstm_checkpoint}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototype file not found at {prototypes_path}")
    if not bam_checkpoint.exists():
        raise FileNotFoundError(f"BAM checkpoint not found at {bam_checkpoint}")

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

    dates_train = splits["dates_train"]
    y_train_series = aligned_labels.reindex(pd.to_datetime(dates_train))
    if y_train_series.isna().any():
        missing_count = int(y_train_series.isna().sum())
        raise ValueError(f"Missing {missing_count} train labels after alignment.")

    x_train = splits["X_train"]
    y_train = y_train_series.astype(int).to_numpy()

    train_dataset = RegimeDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    raw_embeddings, retrieved_embeddings, attention_weights, label_array = extract_bam_embeddings(
        model=model,
        dataloader=train_loader,
        device=device,
    )

    tsne_raw = compute_tsne(raw_embeddings)
    tsne_retrieved = compute_tsne(retrieved_embeddings)

    proto_checkpoint = torch.load(prototypes_path, map_location="cpu")
    if "prototypes" not in proto_checkpoint:
        raise KeyError(f"No 'prototypes' key found in {prototypes_path}")
    prototypes_raw = proto_checkpoint["prototypes"]
    if isinstance(prototypes_raw, torch.Tensor):
        prototypes_raw = prototypes_raw.detach().cpu().numpy()
    else:
        prototypes_raw = np.asarray(prototypes_raw, dtype=float)

    tsne_raw_with_proto = compute_tsne(np.vstack([raw_embeddings, prototypes_raw]))
    proto_left = tsne_raw_with_proto[-3:]

    tsne_retrieved_with_proto = compute_tsne(np.vstack([retrieved_embeddings, prototypes_raw]))
    proto_right = tsne_retrieved_with_proto[-3:]

    prototype_positions = np.hstack([proto_left, proto_right])

    tsne_plot_path = checkpoint_dir / "bam_tsne_comparison.png"
    attention_plot_path = checkpoint_dir / "bam_attention_heatmap.png"
    prototype_distance_path = checkpoint_dir / "bam_prototype_distances.png"

    plot_tsne_comparison(
        tsne_raw=tsne_raw,
        tsne_retrieved=tsne_retrieved,
        labels=label_array,
        prototypes_raw=prototype_positions,
        save_path=tsne_plot_path,
    )
    plot_attention_heatmap(
        attention_weights=attention_weights,
        labels=label_array,
        save_path=attention_plot_path,
    )
    plot_prototype_distances(
        prototypes=prototypes_raw,
        save_path=prototype_distance_path,
    )


if __name__ == "__main__":
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    run_visualisation(device=run_device)
    print("t-SNE visualisation complete")
    print("Plots saved to models/checkpoints/")
    print("Phase 4 complete - ready for Markov smoother")
