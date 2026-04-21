"""Training pipeline for the regime classification LSTM model.

This module loads preprocessed sequence splits and rule-based labels, aligns
labels to split dates, trains the RegimeLSTM classifier with class-balanced
loss, saves the best checkpoint, and reports final test performance.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.labels import align_labels_with_features, load_labels
from data.preprocess import load_splits
from models.lstm import RegimeLSTM, get_model, load_model, save_model


class RegimeDataset(Dataset):
    """Dataset wrapper for regime sequences and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for classes 0, 1, and 2."""
    total_samples = len(y_train)
    n_classes = 3

    class_counts = {cls: int((y_train == cls).sum()) for cls in range(n_classes)}
    weights = []
    for cls in range(n_classes):
        count = class_counts[cls]
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (n_classes * count))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")
    return weight_tensor


def train_one_epoch(
    model: RegimeLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Train model for one epoch and return average loss and accuracy."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(
    model: RegimeLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate model and return average loss and accuracy."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


def train_lstm(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    device: str = "cpu",
) -> dict:
    """Train the RegimeLSTM model and return training history."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(processed_dir)
    labels = load_labels(labels_dir)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    features_df = pd.read_parquet(features_path)
    _, aligned_labels = align_labels_with_features(labels, features_df)

    label_lookup = aligned_labels.copy()

    def labels_for_dates(dates_array: np.ndarray) -> np.ndarray:
        dt_index = pd.to_datetime(dates_array)
        y = label_lookup.reindex(dt_index)
        if y.isna().any():
            missing_count = int(y.isna().sum())
            raise ValueError(
                f"Found {missing_count} split dates without labels after alignment."
            )
        return y.astype(int).to_numpy()

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    y_train = labels_for_dates(splits["dates_train"])
    y_val = labels_for_dates(splits["dates_val"])
    y_test = labels_for_dates(splits["dates_test"])

    train_dataset = RegimeDataset(X_train, y_train)
    val_dataset = RegimeDataset(X_val, y_val)
    test_dataset = RegimeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = int(X_train.shape[-1])
    model = get_model(input_size=input_size, num_classes=3, device=device)

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=5,
    )

    history: dict[str, list[float] | float | str | int] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_checkpoint_path = checkpoint_dir / "lstm_best.pt"
    early_stopping_patience = 10
    epochs_without_improvement = 0
    train_start = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=0.5,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model=model,
                path=best_checkpoint_path,
                metadata={
                    "epoch": epoch + 1,
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                },
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%"
        )
        print(f"Learning rate: {current_lr:.8f}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    best_model = load_model(best_checkpoint_path, device=device)
    test_loss, test_acc = evaluate(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    elapsed = time.time() - train_start
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1f}%")

    history["test_loss"] = float(test_loss)
    history["test_acc"] = float(test_acc)
    history["best_val_loss"] = float(best_val_loss)
    history["best_val_acc"] = float(best_val_acc)
    history["num_epochs"] = int(num_epochs)
    history["batch_size"] = int(batch_size)
    history["learning_rate"] = float(learning_rate)
    history["device"] = device
    history["training_time_sec"] = float(elapsed)
    history["checkpoint_path"] = str(best_checkpoint_path)

    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = train_lstm(device=device)

    print("\nTraining complete")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test accuracy: {results['test_acc']:.2f}%")
    print(f"Checkpoint: {results['checkpoint_path']}")
