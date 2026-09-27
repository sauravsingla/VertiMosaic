from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_at_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, int | float]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
