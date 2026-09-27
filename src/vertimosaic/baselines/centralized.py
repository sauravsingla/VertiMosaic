from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CentralizedBaselineResult:
    """Explicitly non-federated research comparison result."""

    model_name: str
    probabilities: np.ndarray
    label: str = "NON-FEDERATED BASELINE"


def fit_centralized_baseline(
    train_parties: list[np.ndarray],
    y_train: np.ndarray,
    test_parties: list[np.ndarray],
    *,
    model: str = "logistic",
    seed: int = 42,
) -> CentralizedBaselineResult:
    if not train_parties or len(train_parties) != len(test_parties):
        raise ValueError("train and test party lists must be non-empty and aligned")
    x_train = np.column_stack(train_parties)
    x_test = np.column_stack(test_parties)
    if model == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif model == "hist-gbdt":
        estimator = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown centralized baseline: {model}")
    estimator.fit(x_train, np.asarray(y_train).reshape(-1))
    probabilities = estimator.predict_proba(x_test)[:, 1]
    return CentralizedBaselineResult(model_name=model, probabilities=probabilities)


def fit_party_subset_baselines(
    train_by_party: dict[str, np.ndarray],
    y_train: np.ndarray,
    test_by_party: dict[str, np.ndarray],
    *,
    model: str = "logistic",
    seed: int = 42,
    active_party: str = "bank",
) -> dict[str, CentralizedBaselineResult]:
    """Fit every active-plus-passive subset as explicit NON-FEDERATED baselines.

    This helper intentionally concatenates features and therefore exists only for research
    comparison. It must never be described as vertical federated training.
    """
    if set(train_by_party) != set(test_by_party):
        raise ValueError("train and test party mappings must contain the same parties")
    if active_party not in train_by_party:
        raise ValueError(f"active party is missing from baseline mappings: {active_party}")
    passive_names = sorted(name for name in train_by_party if name != active_party)
    output: dict[str, CentralizedBaselineResult] = {}
    for subset_size in range(len(passive_names) + 1):
        for subset in combinations(passive_names, subset_size):
            names = (active_party, *subset)
            key = "+".join(names)
            output[key] = fit_centralized_baseline(
                [train_by_party[name] for name in names],
                y_train,
                [test_by_party[name] for name in names],
                model=model,
                seed=seed,
            )
    return output
