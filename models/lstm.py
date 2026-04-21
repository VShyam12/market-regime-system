"""Bidirectional LSTM model for market regime classification.

The architecture uses two stacked bidirectional LSTM layers to encode
multivariate market sequences, then projects the final timestep representation
into a dense embedding used for both classification logits and downstream
representation learning.
"""

from pathlib import Path

import torch
import torch.nn as nn


class RegimeLSTM(nn.Module):
    """Two-layer bidirectional LSTM for regime classification and embeddings."""

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.embedding_dim = embedding_dim

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            A tuple (logits, embedding):
            - logits: (batch_size, num_classes)
            - embedding: (batch_size, embedding_dim)
        """
        out, _ = self.lstm1(x)
        out = self.dropout(out)

        out, _ = self.lstm2(out)
        out = self.dropout(out)

        last_timestep = out[:, -1, :]
        last_timestep = self.layer_norm(last_timestep)

        embedding = self.embedding_projection(last_timestep)
        logits = self.classifier(embedding)

        return logits, embedding


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    input_size: int = 50,
    num_classes: int = 3,
    device: str = "cpu",
) -> RegimeLSTM:
    """Instantiate and report a RegimeLSTM model."""
    model = RegimeLSTM(input_size=input_size, num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)

    print("RegimeLSTM model created")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: (batch, 60, {input_size})")
    print(f"Output shapes: logits (batch, {num_classes}), embedding (batch, 128)")

    return model


def save_model(model: RegimeLSTM, path: Path, metadata: dict | None = None) -> None:
    """Save model weights, hyperparameters, and optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "dropout": model.dropout_rate,
        "embedding_dim": model.embedding_dim,
    }

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: Path, device: str = "cpu") -> RegimeLSTM:
    """Load a RegimeLSTM model from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    hyperparameters = checkpoint["hyperparameters"]

    model = RegimeLSTM(
        input_size=hyperparameters["input_size"],
        hidden_size=hyperparameters["hidden_size"],
        num_layers=hyperparameters["num_layers"],
        num_classes=hyperparameters["num_classes"],
        dropout=hyperparameters["dropout"],
        embedding_dim=hyperparameters["embedding_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    return model


if __name__ == "__main__":
    model = get_model()

    dummy_input = torch.randn(32, 60, 50)
    logits, embedding = model(dummy_input)

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")

    shape_ok = logits.shape == (32, 3) and embedding.shape == (32, 128)
    print(f"Output shapes correct: {shape_ok}")

    print("\nFull model architecture:")
    print(model)
