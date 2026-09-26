"""Evaluation metrics and uncertainty helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    brier: float
    log_loss: float
    ece: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(mask):
            continue
        ece += np.sum(mask) / total * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def evaluate_binary(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        ece=expected_calibration_error(y, p),
    )


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    *,
    metric: str = "roc_auc",
    replicates: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2 and metric in {"roc_auc", "pr_auc"}:
            continue
        m = evaluate_binary(y[idx], p[idx])
        values.append(float(getattr(m, metric)))
    if not values:
        return (float("nan"), float("nan"))
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))
