from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np


@dataclass
class Party:
    name: str


@dataclass(frozen=True)
class OpaqueSplitReference:
    """Opaque party-local feature/bin reference for a histogram split.

    The numeric threshold is intentionally encapsulated and has no public accessor.
    This is a protocol-boundary abstraction for the in-process research simulator,
    not cryptographic protection against Python introspection.
    """

    feature_ref: int
    bin_ref: int
    _threshold: float = field(repr=False, compare=False)

    def apply(self, party: PassiveParty, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return party._route_with_threshold(indices, self.feature_ref, self._threshold)


class HistogramCandidate(TypedDict):
    """Protocol-visible aggregate statistics plus an opaque local split reference."""

    split_ref: OpaqueSplitReference
    g_left: float
    h_left: float
    g_right: float
    h_right: float
    n_left: int
    n_right: int


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
        feature_indices: np.ndarray | None = None,
    ) -> list[HistogramCandidate]:
        """Compute local split statistics without exposing numeric thresholds."""
        x = self._x[indices]
        out: list[HistogramCandidate] = []
        if len(indices) < 2 * min_samples_leaf:
            return out
        if feature_indices is None:
            features = np.arange(x.shape[1], dtype=int)
        else:
            features = np.asarray(feature_indices, dtype=int).reshape(-1)
            if np.any(features < 0) or np.any(features >= x.shape[1]):
                raise ValueError("feature_indices contain an out-of-range feature")
        for feature_idx in features:
            values = x[:, feature_idx]
            quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, max_bins + 1)[1:-1]))
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
                        "split_ref": OpaqueSplitReference(
                            feature_ref=int(feature_idx),
                            bin_ref=int(threshold_idx),
                            _threshold=float(threshold),
                        ),
                        "g_left": g_left,
                        "h_left": h_left,
                        "g_right": g_right,
                        "h_right": h_right,
                        "n_left": n_left,
                        "n_right": n_right,
                    }
                )
        return out

    def _route_with_threshold(
        self, indices: np.ndarray, feature_idx: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature_idx < 0 or feature_idx >= self.n_features:
            raise ValueError("split feature reference is out of range for this party")
        values = self._x[indices, feature_idx]
        left_mask = values <= threshold
        return indices[left_mask], indices[~left_mask]

    def route_split(
        self, indices: np.ndarray, split_ref: OpaqueSplitReference
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply an opaque split locally; the caller never receives its threshold."""
        return split_ref.apply(self, indices)

    def route(
        self, indices: np.ndarray, feature_idx: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Direct local routing helper retained for controlled tests and diagnostics."""
        return self._route_with_threshold(indices, feature_idx, threshold)


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
