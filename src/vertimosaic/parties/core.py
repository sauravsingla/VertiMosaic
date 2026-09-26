from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Party:
    name: str


@dataclass
class PassiveParty(Party):
    _x: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self._x, dtype=float)
        if x.ndim != 2:
            raise ValueError("party features must be 2D")
        self._x = x

    @property
    def n_rows(self) -> int:
        return self._x.shape[0]

    @property
    def n_features(self) -> int:
        return self._x.shape[1]

    def local_logits(self, weights: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        x = self._x if indices is None else self._x[indices]
        return x @ weights

    def local_gradient(self, residual: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        x = self._x if indices is None else self._x[indices]
        return x.T @ residual / x.shape[0]

    def candidate_histograms(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        max_bins: int,
        min_samples_leaf: int,
    ) -> list[dict[str, float | int]]:
        x = self._x[indices]
        out: list[dict[str, float | int]] = []
        if len(indices) < 2 * min_samples_leaf:
            return out
        for feature_idx in range(x.shape[1]):
            values = x[:, feature_idx]
            quantiles = np.unique(
                np.quantile(values, np.linspace(0.0, 1.0, max_bins + 1)[1:-1])
            )
            for threshold_idx, threshold in enumerate(quantiles):
                left = values <= threshold
                n_left = int(left.sum())
                n_right = len(values) - n_left
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue
                g_left = float(gradients[indices][left].sum())
                h_left = float(hessians[indices][left].sum())
                g_right = float(gradients[indices][~left].sum())
                h_right = float(hessians[indices][~left].sum())
                out.append(
                    {
                        "feature": feature_idx,
                        "threshold_index": threshold_idx,
                        "threshold": float(threshold),
                        "g_left": g_left,
                        "h_left": h_left,
                        "g_right": g_right,
                        "h_right": h_right,
                        "n_left": n_left,
                        "n_right": n_right,
                    }
                )
        return out

    def route(
        self, indices: np.ndarray, feature_idx: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        values = self._x[indices, feature_idx]
        left_mask = values <= threshold
        return indices[left_mask], indices[~left_mask]


@dataclass
class ActiveParty(PassiveParty):
    _y: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        y = np.asarray(self._y, dtype=float).reshape(-1)
        if len(y) != self.n_rows:
            raise ValueError("label vector must match feature rows")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("binary target must contain only 0/1")
        self._y = y

    @property
    def labels(self) -> np.ndarray:
        return self._y
