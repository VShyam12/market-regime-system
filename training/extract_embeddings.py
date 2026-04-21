"""Extract and analyze LSTM embeddings for downstream BAM memory modules.

This module loads the trained LSTM checkpoint, extracts latent embeddings from
training sequences, computes regime prototypes (class means), saves artifacts,
and visualizes the embedding space with PCA.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from models.lstm import load_model
from training.train_lstm import RegimeDataset
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features


REGIME_NAMES = ["Growth", "Transition", "Panic"]
REGIME_COLORS = {0: "green", 1: "orange", 2: "red"}


def extract_all_embeddings(
    model,
    dataloader,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels for all batches in a dataloader."""
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Extracting embeddings"):
            X_batch = X_batch.to(device)

            logits, embeddings = model.forward(X_batch)
            _ = logits  # logits are intentionally unused in this extraction flow.

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    embeddings_np = (
        np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 64), dtype=np.float32)
    )
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)

    return embeddings_np, labels_np


def compute_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    """Compute mean embedding prototypes for each regime class."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array of shape (n_samples, embedding_dim)")

    embedding_dim = embeddings.shape[1]
    prototypes = np.zeros((n_classes, embedding_dim), dtype=np.float32)

    for cls in range(n_classes):
        class_mask = labels == cls
        class_embeddings = embeddings[class_mask]
        count = class_embeddings.shape[0]

        if count == 0:
            print(f"{REGIME_NAMES[cls]} prototype: mean of 0 embeddings (using zeros)")
            prototypes[cls] = np.zeros(embedding_dim, dtype=np.float32)
        else:
            prototypes[cls] = class_embeddings.mean(axis=0)
            print(f"{REGIME_NAMES[cls]} prototype: mean of {count} embeddings")

    d_gt = float(np.linalg.norm(prototypes[0] - prototypes[1]))
    d_gp = float(np.linalg.norm(prototypes[0] - prototypes[2]))
    d_tp = float(np.linalg.norm(prototypes[1] - prototypes[2]))

    print("Inter-prototype L2 distances:")
    print(f"  Growth ↔ Transition: {d_gt:.2f}")
    print(f"  Growth ↔ Panic: {d_gp:.2f}")
    print(f"  Transition ↔ Panic: {d_tp:.2f}")

    return prototypes


def save_embeddings_and_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    checkpoint_dir: Path,
) -> None:
    """Save embedding tensors and regime prototypes to checkpoint artifacts."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = checkpoint_dir / "lstm_embeddings.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"

    torch.save(
        {
            "embeddings": torch.tensor(embeddings),
            "labels": torch.tensor(labels),
        },
        embeddings_path,
    )

    torch.save(
        {
            "prototypes": torch.tensor(prototypes),
            "regime_names": REGIME_NAMES,
        },
        prototypes_path,
    )

    print(f"Saved embeddings to {embeddings_path} with shape {embeddings.shape}")
    print(f"Saved prototypes to {prototypes_path} with shape {prototypes.shape}")


def visualise_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Visualize embedding space and class prototypes in 2D PCA coordinates."""
    if len(embeddings) == 0:
        print("No embeddings available for visualization.")
        return

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)
    proto_2d = pca.transform(prototypes)

    fig = go.Figure()

    for cls in [0, 1, 2]:
        cls_mask = labels == cls
        fig.add_trace(
            go.Scatter(
                x=emb_2d[cls_mask, 0],
                y=emb_2d[cls_mask, 1],
                mode="markers",
                name=REGIME_NAMES[cls],
                marker=dict(color=REGIME_COLORS[cls], size=6, opacity=0.65),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[proto_2d[cls, 0]],
                y=[proto_2d[cls, 1]],
                mode="markers",
                name=f"{REGIME_NAMES[cls]} Prototype",
                marker=dict(
                    symbol="star",
                    color=REGIME_COLORS[cls],
                    size=16,
                    line=dict(color="black", width=2),
                ),
            )
        )

    fig.update_layout(
        title="LSTM embedding space — training data (PCA)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=1000,
        height=700,
        legend_title="Regime",
    )

    if save_path is not None:
        try:
            fig.write_image(str(save_path))
            print(f"Saved embedding visualization to {save_path}")
        except Exception as exc:
            print(f"Could not save embedding visualization PNG: {exc}")

    fig.show()


def run_extraction(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cpu",
) -> dict:
    """Run complete embedding extraction and prototype computation pipeline."""
    checkpoint_path = checkpoint_dir / "lstm_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}")

    model = load_model(checkpoint_path, device=device)

    splits = load_splits(processed_dir)
    labels = load_labels(labels_dir)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    features_df = pd.read_parquet(features_path)
    _, aligned_labels = align_labels_with_features(labels, features_df)

    X_train = splits["X_train"]
    dates_train = pd.to_datetime(splits["dates_train"])
    y_train_series = aligned_labels.reindex(dates_train)

    if y_train_series.isna().any():
        missing_count = int(y_train_series.isna().sum())
        raise ValueError(f"Missing {missing_count} labels for training dates after alignment.")

    y_train = y_train_series.astype(int).to_numpy()

    train_dataset = RegimeDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    embeddings, label_array = extract_all_embeddings(model, train_loader, device)
    prototypes = compute_prototypes(embeddings, label_array, n_classes=3)

    save_embeddings_and_prototypes(embeddings, label_array, prototypes, checkpoint_dir)

    viz_path = checkpoint_dir / "embedding_space_pca.png"
    visualise_embeddings(embeddings, label_array, prototypes, save_path=viz_path)

    return {
        "embeddings": embeddings,
        "labels": label_array,
        "prototypes": prototypes,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_extraction(device=device)

    embeddings = results["embeddings"]
    prototypes = results["prototypes"]

    d_gt = float(np.linalg.norm(prototypes[0] - prototypes[1]))
    d_gp = float(np.linalg.norm(prototypes[0] - prototypes[2]))
    d_tp = float(np.linalg.norm(prototypes[1] - prototypes[2]))

    print(f"Embeddings extracted: shape {embeddings.shape}")
    print(f"Prototypes computed: shape {prototypes.shape}")
    print("Inter-prototype distances:")
    print(f"  Growth ↔ Transition: {d_gt:.2f}")
    print(f"  Growth ↔ Panic: {d_gp:.2f}")
    print(f"  Transition ↔ Panic: {d_tp:.2f}")
    print("Ready for BAM module in Phase 4")
