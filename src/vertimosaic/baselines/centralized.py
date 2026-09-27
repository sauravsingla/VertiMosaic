from __future__ import annotations

from dataclasses import dataclass

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
