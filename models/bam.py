"""Modern Hopfield Network based BAM module for regime memory retrieval.

A Modern Hopfield layer can be interpreted as attention over stored memory
patterns. Given a query embedding, similarities to stored prototypes are scaled
and normalized via softmax, producing retrieval weights. The retrieved vector is
then a weighted sum of memory patterns. This module applies that mechanism on
LSTM embeddings for regime-aware associative retrieval and classification.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm import load_model


class HopfieldBAM(nn.Module):
    """Modern Hopfield memory block with trainable regime prototypes."""

    def __init__(
        self,
        embedding_dim: int = 64,
        n_regimes: int = 3,
        beta: float = 2.0,
        trainable_prototypes: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_regimes = n_regimes
        self.init_beta = beta
        self.trainable_prototypes = trainable_prototypes

        init_patterns = torch.randn(n_regimes, embedding_dim) * 0.02
        self.stored_patterns = nn.Parameter(init_patterns)
        if not trainable_prototypes:
            self.stored_patterns.requires_grad = False

        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))

        self.classifier = nn.Linear(embedding_dim + n_regimes, n_regimes)

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run query retrieval and return logits, retrieved memory, and attention weights."""
        query_norm = F.normalize(query, dim=-1)
        patterns_norm = F.normalize(self.stored_patterns, dim=-1)
        similarities = query_norm @ patterns_norm.t()

        enriched = torch.cat([query, similarities], dim=-1)
        logits = self.classifier(enriched)

        weights = F.softmax(similarities * self.beta, dim=-1)

        retrieved = weights @ self.stored_patterns
        return logits, retrieved, weights

    def initialise_from_prototypes(self, prototypes_path: Path) -> None:
        """Initialize stored patterns from saved LSTM class prototypes."""
        checkpoint = torch.load(prototypes_path, map_location="cpu")
        if "prototypes" not in checkpoint:
            raise KeyError(f"No 'prototypes' key found in {prototypes_path}")

        prototypes = checkpoint["prototypes"]
        if not isinstance(prototypes, torch.Tensor):
            prototypes = torch.tensor(prototypes, dtype=torch.float32)

        if prototypes.shape != self.stored_patterns.data.shape:
            raise ValueError(
                f"Prototype shape mismatch: expected {tuple(self.stored_patterns.data.shape)}, "
                f"got {tuple(prototypes.shape)}"
            )

        self.stored_patterns.data = prototypes.to(self.stored_patterns.data.device).float()
        print("BAM prototypes initialised from LSTM embeddings")

    def get_regime_attention(self, query: torch.Tensor) -> torch.Tensor:
        """Return only regime attention weights for interpretability."""
        _, _, weights = self.forward(query)
        return weights


class LSTMBAMModel(nn.Module):
    """Frozen LSTM encoder followed by trainable Hopfield BAM head."""

    def __init__(
        self,
        lstm_checkpoint: Path,
        prototypes_path: Path,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device = device

        self.lstm = load_model(lstm_checkpoint, device=device)
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.bam = HopfieldBAM(embedding_dim=64, n_regimes=3)
        self.bam.initialise_from_prototypes(prototypes_path)

        self.to(device)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run frozen LSTM encoding then BAM retrieval/classification."""
        _, embedding = self.lstm(x)
        logits, retrieved, weights = self.bam(embedding)
        return logits, embedding, retrieved, weights

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return BAM trainable parameters and report frozen/trainable counts."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        frozen_params = [p for p in self.parameters() if not p.requires_grad]

        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = sum(p.numel() for p in frozen_params)

        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Frozen parameters: {frozen_count:,}")

        return list(self.bam.parameters())


def save_bam_model(model: LSTMBAMModel, path: Path, metadata: dict | None = None) -> None:
    """Save BAM-only weights and optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "bam_state_dict": model.bam.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Saved BAM model to {path}")


def load_bam_model(
    path: Path,
    lstm_checkpoint: Path,
    prototypes_path: Path,
    device: str = "cpu",
) -> LSTMBAMModel:
    """Load BAM-only checkpoint into a reconstructed LSTMBAMModel."""
    model = LSTMBAMModel(
        lstm_checkpoint=lstm_checkpoint,
        prototypes_path=prototypes_path,
        device=device,
    )

    checkpoint = torch.load(path, map_location=device)
    if "bam_state_dict" not in checkpoint:
        raise KeyError(f"No 'bam_state_dict' found in {path}")

    model.bam.load_state_dict(checkpoint["bam_state_dict"])
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_ckpt = Path("models/checkpoints/lstm_best.pt")
    proto_ckpt = Path("models/checkpoints/lstm_prototypes.pt")

    model = LSTMBAMModel(
        lstm_checkpoint=lstm_ckpt,
        prototypes_path=proto_ckpt,
        device=device,
    )

    dummy_x = torch.randn(32, 60, 50, device=device)
    logits, embedding, retrieved, weights = model(dummy_x)

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Retrieved shape: {tuple(retrieved.shape)}")
    print(f"Attention weights shape: {tuple(weights.shape)}")

    bam_params = model.get_trainable_params()
    print(f"BAM parameter tensors: {len(bam_params)}")
    print(f"Stored patterns shape: {tuple(model.bam.stored_patterns.shape)}")
    print("LSTMBAMModel forward pass successful")
