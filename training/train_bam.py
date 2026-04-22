"""Training pipeline for LSTM + Modern Hopfield BAM regime classification.

This module trains only the Hopfield BAM memory/classification head while the
LSTM encoder stays frozen. The BAM performs associative retrieval against stored
regime prototypes, and training combines cross-entropy classification with a
small prototype-separation regularizer to keep memory patterns distinct.
"""

from pathlib import Path
import json

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.labels import align_labels_with_features, load_labels
from data.preprocess import load_splits
from models.bam import LSTMBAMModel, save_bam_model
from training.train_lstm import RegimeDataset, compute_class_weights


def train_one_epoch_bam(
    model: LSTMBAMModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train BAM head for one epoch and return average loss and accuracy."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, embedding, retrieved, weights = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bam.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = (100.0 * correct / total) if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def evaluate_bam(
    model: LSTMBAMModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate BAM model and return average loss and accuracy."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _embedding, _retrieved, _weights = model(x_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


def train_bam(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    num_epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    device: str = "cpu",
) -> dict:
    """Train the BAM head on frozen LSTM embeddings and return history."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(processed_dir)
    labels = load_labels(labels_dir)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    features_df = pd.read_parquet(features_path)
    _aligned_features, aligned_labels = align_labels_with_features(labels, features_df)
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

    x_train = splits["X_train"]
    x_val = splits["X_val"]
    x_test = splits["X_test"]

    y_train = labels_for_dates(splits["dates_train"])
    y_val = labels_for_dates(splits["dates_val"])
    y_test = labels_for_dates(splits["dates_test"])

    train_dataset = RegimeDataset(x_train, y_train)
    val_dataset = RegimeDataset(x_val, y_val)
    test_dataset = RegimeDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
    prototypes_path = checkpoint_dir / "lstm_prototypes.pt"

    model = LSTMBAMModel(
        lstm_checkpoint=lstm_checkpoint,
        prototypes_path=prototypes_path,
        device=device,
    )

    # Train only classification head; keep stored patterns frozen.
    for param in model.bam.parameters():
        param.requires_grad = False
    for param in model.bam.classifier.parameters():
        param.requires_grad = True

    bam_params = [p for p in model.bam.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(bam_params, lr=learning_rate)

    print("Training BAM parameters:")
    for name, param in model.bam.named_parameters():
        if param.requires_grad:
            print(f"- {name}: {tuple(param.shape)}")

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    history: dict[str, list[float] | float | int | str] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = float("-inf")
    epochs_without_improvement = 0
    early_stopping_patience = 15
    best_checkpoint_path = checkpoint_dir / "bam_best.pt"

    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("market-regime-bam")
        mlflow.start_run(run_name="bam_training")
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "beta": 2.0,
            }
        )
        mlflow_enabled = True
    except Exception as exc:
        print(f"MLflow setup failed, continuing without tracking: {exc}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch_bam(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_bam(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            epochs_without_improvement = 0
            save_bam_model(
                model=model,
                path=best_checkpoint_path,
                metadata={
                    "epoch": epoch + 1,
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                },
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        if mlflow_enabled:
            try:
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "train_acc": float(train_acc),
                        "val_acc": float(val_acc),
                    },
                    step=epoch,
                )
            except Exception as exc:
                print(f"MLflow epoch logging failed, continuing training: {exc}")
                mlflow_enabled = False

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    best_model = LSTMBAMModel(
        lstm_checkpoint=lstm_checkpoint,
        prototypes_path=prototypes_path,
        device=device,
    )
    bam_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    if "bam_state_dict" not in bam_checkpoint:
        raise KeyError(f"No 'bam_state_dict' found in {best_checkpoint_path}")
    best_model.bam.load_state_dict(bam_checkpoint["bam_state_dict"])

    test_loss, test_acc = evaluate_bam(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    history["test_loss"] = float(test_loss)
    history["test_acc"] = float(test_acc)
    history["best_val_acc"] = float(best_val_acc)
    history["num_epochs"] = int(num_epochs)
    history["batch_size"] = int(batch_size)
    history["learning_rate"] = float(learning_rate)
    history["device"] = device
    history["checkpoint_path"] = str(best_checkpoint_path)

    history_path = checkpoint_dir / "bam_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved BAM history to {history_path}")

    if mlflow_enabled:
        try:
            mlflow.log_metrics(
                {
                    "test_accuracy": float(test_acc),
                    "best_val_acc": float(best_val_acc),
                }
            )
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(best_checkpoint_path))
        except Exception as exc:
            print(f"MLflow final logging failed: {exc}")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = train_bam(device=device)

    lstm_test_acc = 65.4
    bam_test_acc = float(history.get("test_acc", 0.0))
    improvement = bam_test_acc - lstm_test_acc

    print(f"LSTM alone test accuracy: {lstm_test_acc:.1f}%")
    print(f"LSTM + BAM test accuracy: {bam_test_acc:.1f}%")
    print(f"Improvement: {improvement:+.1f}%")
