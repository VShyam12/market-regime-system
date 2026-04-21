"""Validate rule-based regime labels with a Gaussian HMM.

The validation workflow fits an unsupervised Gaussian HMM on selected market
features, maps hidden states to the rule-based regime labels using majority
voting, measures agreement, and summarizes the inferred state dynamics.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


HMM_FEATURE_COLS = [
    "SPY_return_1d",
    "SPY_vol_20d",
    "vix_level",
    "bond_equity_ratio",
    "SPY_return_21d",
]

REGIME_NAMES = {0: "Growth", 1: "Transition", 2: "Panic"}


def _select_hmm_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature subset required for HMM fitting and prediction."""
    missing = [col for col in HMM_FEATURE_COLS if col not in features_df.columns]
    if missing:
        raise KeyError(f"Missing required HMM feature columns: {missing}")

    return features_df.loc[:, HMM_FEATURE_COLS].copy()


def fit_hmm(
    features_df: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 100,
) -> GaussianHMM:
    """Fit a Gaussian HMM on the selected market features."""
    hmm_features = _select_hmm_features(features_df)
    fit_features = hmm_features.dropna(axis=0, how="any")

    if fit_features.empty:
        raise ValueError("No complete rows available to fit the HMM after dropping NaNs.")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(fit_features.to_numpy())

    log_likelihood = model.score(fit_features.to_numpy())
    print(f"HMM fitted on {len(fit_features)} rows")
    print(f"Model log likelihood: {log_likelihood:.3f}")

    return model


def get_hmm_states(model: GaussianHMM, features_df: pd.DataFrame) -> pd.Series:
    """Predict HMM hidden states for all dates in the feature matrix."""
    hmm_features = _select_hmm_features(features_df)
    filled_features = hmm_features.ffill().bfill()

    if filled_features.isna().any().any():
        raise ValueError("Unable to fill all NaNs in HMM features before prediction.")

    states = model.predict(filled_features.to_numpy())
    return pd.Series(states.astype(int), index=features_df.index, name="hmm_state")


def align_hmm_to_rules(hmm_states: pd.Series, rule_labels: pd.Series) -> dict:
    """Map HMM states to regime labels using majority voting."""
    aligned = pd.concat(
        [
            hmm_states.rename("hmm_state"),
            rule_labels.rename("rule_label"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlapping dates found between HMM states and rule labels.")

    mapping: dict[int, int] = {}
    for state in sorted(aligned["hmm_state"].astype(int).unique()):
        state_rows = aligned.loc[aligned["hmm_state"].astype(int) == state, "rule_label"].astype(int)
        majority_label = int(state_rows.value_counts().idxmax())
        mapping[int(state)] = majority_label

    printable = {state: REGIME_NAMES.get(label, str(label)) for state, label in mapping.items()}
    print("HMM state to regime mapping:")
    for state, regime_name in printable.items():
        print(f"- State {state} -> {regime_name}")

    return mapping


def compute_agreement(
    hmm_states: pd.Series,
    rule_labels: pd.Series,
    state_mapping: dict,
) -> float:
    """Compute agreement percentage between mapped HMM states and rule labels."""
    aligned = pd.concat(
        [
            hmm_states.rename("hmm_state"),
            rule_labels.rename("rule_label"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlapping dates found when computing agreement.")

    mapped_hmm_labels = aligned["hmm_state"].astype(int).map(state_mapping)
    valid_mask = mapped_hmm_labels.notna()
    aligned = aligned.loc[valid_mask].copy()
    mapped_hmm_labels = mapped_hmm_labels.loc[valid_mask].astype(int)
    rule_values = aligned["rule_label"].astype(int)

    agreement = (mapped_hmm_labels == rule_values).mean() * 100.0
    print(f"Agreement score: {agreement:.2f}%")

    confusion = pd.crosstab(
        pd.Series(mapped_hmm_labels, name="HMM mapped"),
        pd.Series(rule_values, name="Rule labels"),
        dropna=False,
    )
    print("Confusion matrix:")
    print(confusion)

    return float(agreement)


def analyse_hmm_states(
    model: GaussianHMM,
    feature_cols: list,
    state_mapping: dict,
) -> pd.DataFrame:
    """Summarize mean feature values for each HMM state."""
    feature_names = list(feature_cols)
    state_means = pd.DataFrame(model.means_, columns=feature_names)
    state_means.insert(0, "hmm_state", range(len(state_means)))
    state_means["regime"] = state_means["hmm_state"].map(
        lambda state: REGIME_NAMES.get(state_mapping.get(int(state), -1), f"State {state}")
    )

    display_cols = ["hmm_state", "regime"] + [col for col in feature_names if col in state_means.columns]
    analysis_df = state_means.loc[:, display_cols].copy()

    print("HMM state feature means:")
    print(analysis_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return analysis_df


def extract_transition_matrix(model: GaussianHMM, state_mapping: dict) -> pd.DataFrame:
    """Reorder and save the HMM transition matrix in regime order."""
    regime_order = [0, 1, 2]
    regime_names = [REGIME_NAMES[label] for label in regime_order]

    regime_to_state: dict[int, int] = {}
    for state, regime_label in sorted(state_mapping.items()):
        if regime_label in regime_to_state:
            continue
        regime_to_state[int(regime_label)] = int(state)

    missing_regimes = [label for label in regime_order if label not in regime_to_state]
    if missing_regimes:
        raise ValueError(
            f"Unable to reorder transition matrix because these regime labels are missing from the mapping: {missing_regimes}"
        )

    state_order = [regime_to_state[label] for label in regime_order]
    transition_matrix = pd.DataFrame(
        model.transmat_[np.ix_(state_order, state_order)],
        index=regime_names,
        columns=regime_names,
    )

    print("Transition matrix:")
    print(transition_matrix.to_string(float_format=lambda value: f"{value:.3f}"))

    output_path = Path("data/labels/hmm_transition_matrix.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transition_matrix.to_csv(output_path)
    print(f"Saved transition matrix to {output_path}")

    return transition_matrix


def run_validation(
    processed_dir: Path = Path("data/processed"),
    labels_dir: Path = Path("data/labels"),
) -> dict:
    """Run the full HMM validation pipeline and return all intermediate outputs."""
    from data.labels import load_labels

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    features_df = pd.read_parquet(features_path)
    rule_labels = load_labels(labels_dir)

    model = fit_hmm(features_df)
    hmm_states = get_hmm_states(model, features_df)

    common_dates = hmm_states.index.intersection(rule_labels.index)
    hmm_states_aligned = hmm_states.loc[common_dates]
    rule_labels_aligned = rule_labels.loc[common_dates]

    state_mapping = align_hmm_to_rules(hmm_states_aligned, rule_labels_aligned)
    agreement = compute_agreement(hmm_states_aligned, rule_labels_aligned, state_mapping)
    analysis_df = analyse_hmm_states(model, HMM_FEATURE_COLS, state_mapping)
    transition_matrix = extract_transition_matrix(model, state_mapping)

    results = {
        "model": model,
        "hmm_states": hmm_states,
        "hmm_states_aligned": hmm_states_aligned,
        "rule_labels": rule_labels,
        "rule_labels_aligned": rule_labels_aligned,
        "state_mapping": state_mapping,
        "agreement": agreement,
        "analysis_df": analysis_df,
        "transition_matrix": transition_matrix,
    }

    return results


if __name__ == "__main__":
    try:
        results = run_validation()
        agreement = results["agreement"]

        print("HMM validation complete")
        print(f"Agreement score: {agreement:.2f}%")
        print("Transition matrix saved to data/labels/hmm_transition_matrix.csv")
        if agreement > 60:
            print("If agreement > 60%, labels are validated and ready for training")
        else:
            print("If agreement > 60%, labels are validated and ready for training")
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise
