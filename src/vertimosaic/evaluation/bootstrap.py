from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def _metric(name: str, y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    if name == "roc_auc":
        return float(roc_auc_score(y, p))
    if name == "pr_auc":
        return float(average_precision_score(y, p))
    if name == "brier":
        return float(brier_score_loss(y, p))
    if name == "f1":
        return float(f1_score(y, p >= threshold, zero_division=0))
    raise ValueError(f"unsupported metric: {name}")


def _interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap deterministic 95% intervals on held-out test predictions."""
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal length")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    names = ("roc_auc", "pr_auc", "brier", "f1")
    samples: dict[str, list[float]] = {name: [] for name in names}
    n = len(y)
    for _ in range(replicates):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_b = p[idx]
        single_class = np.unique(y_b).size < 2
        for name in names:
            if single_class and name in {"roc_auc", "pr_auc"}:
                continue
            samples[name].append(_metric(name, y_b, p_b, threshold))
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        low, high = _interval(samples[name])
        result[name] = {
            "estimate": _metric(name, y, p, threshold),
            "lower": low,
            "upper": high,
            "valid_replicates": len(samples[name]),
        }
    return result


def paired_bootstrap_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    *,
    metric: str = "roc_auc",
    threshold: float = 0.5,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Return a paired bootstrap interval for metric(A) minus metric(B)."""
    y = np.asarray(y_true).reshape(-1)
    a = np.asarray(probabilities_a, dtype=float).reshape(-1)
    b = np.asarray(probabilities_b, dtype=float).reshape(-1)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("paired predictions must have equal length")
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = len(y)
    for _ in range(replicates):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if metric in {"roc_auc", "pr_auc"} and np.unique(y_b).size < 2:
            continue
        deltas.append(
            _metric(metric, y_b, a[idx], threshold)
            - _metric(metric, y_b, b[idx], threshold)
        )
    low, high = _interval(deltas)
    point = _metric(metric, y, a, threshold) - _metric(metric, y, b, threshold)
    return {
        "metric": metric,
        "delta": float(point),
        "lower": low,
        "upper": high,
        "valid_replicates": len(deltas),
    }


def select_f1_threshold(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid_size: int = 201,
) -> float:
    """Choose an F1 threshold from validation data only."""
    y = np.asarray(y_validation).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("validation labels and probabilities must have equal length")
    grid = np.linspace(0.0, 1.0, grid_size)
    scores = np.asarray([f1_score(y, p >= threshold, zero_division=0) for threshold in grid])
    return float(grid[int(np.argmax(scores))])
