"""Hidden Markov Model utilities for regime-sequence smoothing.

This module estimates Markov parameters from labeled data and model outputs,
then applies Viterbi decoding to smooth noisy regime predictions.

Model components:
- Transition matrix A: P(s_t = j | s_{t-1} = i)
- Emission matrix B: P(o_t = k | s_t = j)
- Initial distribution pi: P(s_0 = j)

Viterbi decoding finds the most likely hidden regime path given observations.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np

from data.labels import load_labels


STATE_NAMES = ["Growth", "Transition", "Panic"]


def _state_name(idx: int) -> str:
    if 0 <= idx < len(STATE_NAMES):
        return STATE_NAMES[idx]
    return f"State {idx}"


def _validate_state_array(values: np.ndarray, n_states: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(arr < 0) or np.any(arr >= n_states):
        bad = arr[(arr < 0) | (arr >= n_states)][:10]
        raise ValueError(f"{name} contains invalid state ids: {bad}")
    return arr


def estimate_transition_matrix(
    labels: np.ndarray,
    n_states: int = 3,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Estimate transition probabilities from consecutive label pairs."""
    label_arr = _validate_state_array(labels, n_states, "labels")
    if label_arr.size < 2:
        raise ValueError("Need at least 2 labels to estimate transitions.")

    # Transition count matrix C[i, j] counts i -> j transitions.
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(label_arr.size - 1):
        i = int(label_arr[t])
        j = int(label_arr[t + 1])
        counts[i, j] += 1.0

    # Laplace smoothing avoids zero probabilities.
    counts += float(smoothing)

    # Row-normalized transition matrix A where rows sum to 1.
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)

    print("Transition matrix (A):")
    print(np.array2string(transition_matrix, formatter={"float_kind": lambda x: f"{x:.4f}"}))

    diagonal = np.diag(transition_matrix)
    for s in range(n_states):
        stay_prob = float(diagonal[s])
        denom = max(1e-10, 1.0 - stay_prob)
        dwell_time = 1.0 / denom
        print(f"{_state_name(s)} mean dwell: {dwell_time:.1f} days")

    return transition_matrix


def estimate_emission_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    n_states: int = 3,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Estimate emission probabilities P(prediction | true_state)."""
    true_states = _validate_state_array(labels, n_states, "labels")
    pred_states = _validate_state_array(predictions, n_states, "predictions")

    if true_states.size != pred_states.size:
        raise ValueError("labels and predictions must have the same length.")

    # Emission count matrix E[j, k] counts true state j emitting observation k.
    counts = np.zeros((n_states, n_states), dtype=float)
    for j, k in zip(true_states, pred_states):
        counts[int(j), int(k)] += 1.0

    counts += float(smoothing)
    emission_matrix = counts / counts.sum(axis=1, keepdims=True)

    print("Emission matrix (B):")
    print(np.array2string(emission_matrix, formatter={"float_kind": lambda x: f"{x:.4f}"}))

    return emission_matrix


def estimate_initial_distribution(
    labels: np.ndarray,
    n_states: int = 3,
) -> np.ndarray:
    """Estimate initial regime distribution from label frequencies."""
    label_arr = _validate_state_array(labels, n_states, "labels")

    counts = np.bincount(label_arr, minlength=n_states).astype(float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot estimate initial distribution from empty labels.")

    initial_dist = counts / total
    print("Initial distribution (pi):")
    print(np.array2string(initial_dist, formatter={"float_kind": lambda x: f"{x:.4f}"}))

    return initial_dist


def viterbi_decode(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_dist: np.ndarray,
) -> np.ndarray:
    """Decode most likely hidden state sequence using Viterbi on hard observations."""
    obs = np.asarray(observations, dtype=int).reshape(-1)
    if obs.size == 0:
        return np.array([], dtype=int)

    n_states = int(initial_dist.shape[0])
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape must be (n_states, n_states)")
    if emission_matrix.shape != (n_states, n_states):
        raise ValueError("emission_matrix shape must be (n_states, n_states)")

    log_trans = np.log(np.asarray(transition_matrix, dtype=float) + 1e-10)
    log_emit = np.log(np.asarray(emission_matrix, dtype=float) + 1e-10)
    log_init = np.log(np.asarray(initial_dist, dtype=float) + 1e-10)

    t_len = obs.size
    delta = np.full((t_len, n_states), -np.inf, dtype=float)
    psi = np.zeros((t_len, n_states), dtype=int)

    delta[0] = log_init + log_emit[:, obs[0]]

    for t in range(1, t_len):
        for j in range(n_states):
            scores = delta[t - 1] + log_trans[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = float(np.max(scores)) + log_emit[j, obs[t]]

    decoded = np.zeros(t_len, dtype=int)
    decoded[-1] = int(np.argmax(delta[-1]))

    for t in range(t_len - 2, -1, -1):
        decoded[t] = psi[t + 1, decoded[t + 1]]

    return decoded


def smooth_predictions(
    raw_predictions: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_dist: np.ndarray,
) -> np.ndarray:
    """Apply Viterbi smoothing to hard prediction sequence."""
    raw = np.asarray(raw_predictions, dtype=int).reshape(-1)
    smoothed = viterbi_decode(raw, transition_matrix, emission_matrix, initial_dist)

    changed = int(np.sum(raw != smoothed))
    raw_changes = int(np.sum(raw[1:] != raw[:-1])) if raw.size > 1 else 0
    smooth_changes = int(np.sum(smoothed[1:] != smoothed[:-1])) if smoothed.size > 1 else 0

    print(f"Predictions changed after smoothing: {changed}/{raw.size}")
    print(f"Regime changes before smoothing: {raw_changes}")
    print(f"Regime changes after smoothing: {smooth_changes}")

    return smoothed


def viterbi_decode_with_probs(
    prob_sequence: np.ndarray,
    transition_matrix: np.ndarray,
    initial_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode hidden states with Viterbi using soft class probabilities as emissions."""
    probs = np.asarray(prob_sequence, dtype=float)
    if probs.ndim != 2:
        raise ValueError("prob_sequence must have shape (T, n_states)")

    t_len, n_states = probs.shape
    if t_len == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape must be (n_states, n_states)")
    if initial_dist.shape != (n_states,):
        raise ValueError("initial_dist shape must be (n_states,)")

    log_trans = np.log(np.asarray(transition_matrix, dtype=float) + 1e-10)
    log_init = np.log(np.asarray(initial_dist, dtype=float) + 1e-10)
    log_emit_score = np.log(probs + 1e-10)

    delta = np.full((t_len, n_states), -np.inf, dtype=float)
    psi = np.zeros((t_len, n_states), dtype=int)

    delta[0] = log_init + log_emit_score[0]

    for t in range(1, t_len):
        for j in range(n_states):
            scores = delta[t - 1] + log_trans[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = float(np.max(scores)) + log_emit_score[t, j]

    decoded = np.zeros(t_len, dtype=int)
    decoded[-1] = int(np.argmax(delta[-1]))

    for t in range(t_len - 2, -1, -1):
        decoded[t] = psi[t + 1, decoded[t + 1]]

    confidence_scores = probs[np.arange(t_len), decoded]
    return decoded, confidence_scores


def save_markov_params(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_dist: np.ndarray,
    save_path: Path,
) -> None:
    """Save Markov parameters to NPZ and transition matrix to CSV."""
    save_path.mkdir(parents=True, exist_ok=True)

    npz_path = save_path / "markov_params.npz"
    csv_path = save_path / "markov_transition_matrix.csv"

    np.savez(
        npz_path,
        transition_matrix=np.asarray(transition_matrix, dtype=float),
        emission_matrix=np.asarray(emission_matrix, dtype=float),
        initial_dist=np.asarray(initial_dist, dtype=float),
    )
    np.savetxt(csv_path, np.asarray(transition_matrix, dtype=float), delimiter=",", fmt="%.6f")

    print(f"Saved Markov parameters to {npz_path}")
    print(f"Saved transition matrix CSV to {csv_path}")


def load_markov_params(load_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Markov parameters from NPZ file."""
    npz_path = load_path / "markov_params.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Markov parameter file not found at {npz_path}")

    data = np.load(npz_path)
    transition_matrix = np.asarray(data["transition_matrix"], dtype=float)
    emission_matrix = np.asarray(data["emission_matrix"], dtype=float)
    initial_dist = np.asarray(data["initial_dist"], dtype=float)

    return transition_matrix, emission_matrix, initial_dist


def _ensure_bam_train_outputs(
    project_root: Path,
    checkpoint_dir: Path,
    labels_dir: Path,
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate BAM train predictions/probabilities via a subprocess inference run."""
    pred_path = checkpoint_dir / "bam_train_predictions.npy"
    prob_path = checkpoint_dir / "bam_train_probabilities.npy"

    if pred_path.exists() and prob_path.exists():
        return np.load(pred_path, allow_pickle=False), np.load(prob_path, allow_pickle=False)

    script = """
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.bam import load_bam_model
from training.train_lstm import RegimeDataset
from data.preprocess import load_splits
from data.labels import load_labels, align_labels_with_features

checkpoint_dir = Path('models/checkpoints')
processed_dir = Path('data/processed')
labels_dir = Path('data/labels')

model = load_bam_model(
    path=checkpoint_dir / 'bam_best.pt',
    lstm_checkpoint=checkpoint_dir / 'lstm_best.pt',
    prototypes_path=checkpoint_dir / 'lstm_prototypes.pt',
    device='cpu',
)
model.eval()

splits = load_splits(processed_dir)
labels = load_labels(labels_dir)
features_df = pd.read_parquet(processed_dir / 'features.parquet')
_, aligned_labels = align_labels_with_features(labels, features_df)

dates_train = pd.to_datetime(splits['dates_train'])
y_train = aligned_labels.reindex(dates_train).astype(int).to_numpy()
x_train = splits['X_train']

loader = DataLoader(RegimeDataset(x_train, y_train), batch_size=256, shuffle=False)

preds_list = []
probs_list = []
with torch.no_grad():
    for xb, _yb in loader:
        logits, _emb, _ret, _w = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_list.append(preds.cpu().numpy())
        probs_list.append(probs.cpu().numpy())

preds = np.concatenate(preds_list, axis=0) if preds_list else np.array([], dtype=int)
probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3), dtype=float)

np.save(checkpoint_dir / 'bam_train_predictions.npy', preds)
np.save(checkpoint_dir / 'bam_train_probabilities.npy', probs)
print('Saved BAM train predictions/probabilities')
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to run BAM inference subprocess for training predictions.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not pred_path.exists() or not prob_path.exists():
        raise FileNotFoundError("BAM train output files were not created by subprocess run.")

    return np.load(pred_path, allow_pickle=False), np.load(prob_path, allow_pickle=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    labels_dir = root / "data" / "labels"
    processed_dir = root / "data" / "processed"
    checkpoint_dir = root / "models" / "checkpoints"

    labels_series = load_labels(labels_dir)

    dates_train_path = processed_dir / "dates_train.npy"
    if not dates_train_path.exists():
        raise FileNotFoundError(f"Train date file not found at {dates_train_path}")

    dates_train = np.load(dates_train_path, allow_pickle=False)
    train_labels_series = labels_series.reindex(dates_train)
    if train_labels_series.isna().any():
        missing_count = int(train_labels_series.isna().sum())
        raise ValueError(f"Missing {missing_count} train labels after alignment.")

    train_labels = train_labels_series.astype(int).to_numpy()

    bam_predictions, bam_probabilities = _ensure_bam_train_outputs(
        project_root=root,
        checkpoint_dir=checkpoint_dir,
        labels_dir=labels_dir,
        processed_dir=processed_dir,
    )

    if bam_predictions.shape[0] != train_labels.shape[0]:
        raise ValueError(
            "Prediction/label length mismatch: "
            f"predictions={bam_predictions.shape[0]}, labels={train_labels.shape[0]}"
        )

    transition_matrix = estimate_transition_matrix(train_labels)
    emission_matrix = estimate_emission_matrix(train_labels, bam_predictions)
    initial_dist = estimate_initial_distribution(train_labels)

    example_obs = bam_predictions[:10]
    decoded_example = viterbi_decode(
        observations=example_obs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        initial_dist=initial_dist,
    )
    print(f"Viterbi example (10 steps) input:  {example_obs}")
    print(f"Viterbi example (10 steps) output: {decoded_example}")

    decoded_soft, conf = viterbi_decode_with_probs(
        prob_sequence=bam_probabilities[:10],
        transition_matrix=transition_matrix,
        initial_dist=initial_dist,
    )
    print(f"Soft-Viterbi example states:       {decoded_soft}")
    print(f"Soft-Viterbi confidences:          {np.round(conf, 4)}")

    save_markov_params(
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        initial_dist=initial_dist,
        save_path=checkpoint_dir,
    )

    print("Markov parameters estimated and saved")
    print("Ready for Phase 5 Step 2: Viterbi smoothing")
