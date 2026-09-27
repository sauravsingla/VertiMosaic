from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import uuid4

import numpy as np


@dataclass
class Party:
    name: str


@dataclass(frozen=True)
class OpaqueSplitReference:
    """Coordinator-visible opaque reference for a party-local histogram split.

    The reference contains only an opaque routing-state token and local slot indices.
    It deliberately carries no numeric split threshold, feature name, or raw feature
    value. The owning party resolves the token against its private routing-state store.
    """

    state_ref: str
    feature_ref: int
    bin_ref: int


@dataclass(frozen=True)
class HistogramRoutingState:
    """Immutable party-local threshold state learned from training rows only."""

    _thresholds: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    max_bins: int

    def __post_init__(self) -> None:
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        normalized: list[np.ndarray] = []
        for values in self._thresholds:
            array = np.asarray(values, dtype=float).reshape(-1).copy()
            array.setflags(write=False)
            normalized.append(array)
        object.__setattr__(self, "_thresholds", tuple(normalized))

    @property
    def n_features(self) -> int:
        return len(self._thresholds)

    def _threshold_for(self, split_ref: OpaqueSplitReference) -> float:
        feature_ref = split_ref.feature_ref
        bin_ref = split_ref.bin_ref
        if feature_ref < 0 or feature_ref >= len(self._thresholds):
            raise ValueError("split feature reference is out of range for routing state")
        feature_thresholds = self._thresholds[feature_ref]
        if bin_ref < 0 or bin_ref >= len(feature_thresholds):
            raise ValueError("split bin reference is out of range for routing state")
        return float(feature_thresholds[bin_ref])


# Simulation of party-owned model state. Only derived thresholds are stored here; raw
# feature rows never enter this registry. The coordinating model sees only state_ref.
_PARTY_HISTOGRAM_ROUTING_STATES: dict[str, dict[str, HistogramRoutingState]] = {}


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
    _histogram_bins: np.ndarray | None = field(default=None, init=False, repr=False)
    _histogram_thresholds: dict[int, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _histogram_max_bins: int | None = field(default=None, init=False, repr=False)
    _histogram_state_ref: str | None = field(default=None, init=False, repr=False)

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

    @property
    def histogram_bins_ready(self) -> bool:
        """Whether party-local histogram bins have been fitted for this matrix."""
        return self._histogram_bins is not None and self._histogram_max_bins is not None

    def local_logits(self, weights: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        x = self._x if indices is None else self._x[indices]
        return x @ weights

    def local_gradient(self, residual: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        x = self._x if indices is None else self._x[indices]
        return x.T @ residual / x.shape[0]

    def prepare_histogram_bins(self, max_bins: int) -> None:
        """Fit and retain party-local quantile bins once for histogram tree training.

        Thresholds and binned row values remain party-local. Candidate generation then
        aggregates gradients/Hessians by retained bin codes instead of recomputing raw
        value quantiles at every tree node.
        """
        if max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        binned = np.zeros(self._x.shape, dtype=np.int32)
        thresholds: dict[int, np.ndarray] = {}
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        for feature_idx in range(self.n_features):
            values = self._x[:, feature_idx]
            finite = values[np.isfinite(values)]
            feature_thresholds = (
                np.unique(np.quantile(finite, quantiles)).astype(float)
                if finite.size and quantiles.size
                else np.empty(0, dtype=float)
            )
            thresholds[feature_idx] = feature_thresholds
            codes = np.searchsorted(feature_thresholds, values, side="left").astype(np.int32)
            if np.any(~np.isfinite(values)):
                codes[~np.isfinite(values)] = len(feature_thresholds)
            binned[:, feature_idx] = codes

        state = HistogramRoutingState(
            tuple(thresholds[index] for index in range(self.n_features)),
            max_bins=max_bins,
        )
        state_ref = uuid4().hex
        _PARTY_HISTOGRAM_ROUTING_STATES.setdefault(self.name, {})[state_ref] = state

        self._histogram_bins = binned
        self._histogram_thresholds = thresholds
        self._histogram_max_bins = max_bins
        self._histogram_state_ref = state_ref

    def _ensure_histogram_bins(self, max_bins: int) -> None:
        if self._histogram_bins is None or self._histogram_max_bins != max_bins:
            self.prepare_histogram_bins(max_bins)

    def _routing_state_for(self, split_ref: OpaqueSplitReference) -> HistogramRoutingState:
        states = _PARTY_HISTOGRAM_ROUTING_STATES.get(self.name, {})
        state = states.get(split_ref.state_ref)
        if state is None:
            raise ValueError("opaque split reference does not belong to this party")
        if state.n_features != self.n_features:
            raise ValueError("routing state feature width does not match this party")
        return state

    def candidate_histograms(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        max_bins: int,
        min_samples_leaf: int,
        feature_indices: np.ndarray | None = None,
    ) -> list[HistogramCandidate]:
        """Compute local split statistics from retained bins without exposing thresholds."""
        out: list[HistogramCandidate] = []
        if len(indices) < 2 * min_samples_leaf:
            return out
        self._ensure_histogram_bins(max_bins)
        if self._histogram_bins is None or self._histogram_state_ref is None:
            raise RuntimeError("party-local histogram bins were not prepared")
        if feature_indices is None:
            features = np.arange(self.n_features, dtype=int)
        else:
            features = np.asarray(feature_indices, dtype=int).reshape(-1)
            if np.any(features < 0) or np.any(features >= self.n_features):
                raise ValueError("feature_indices contain an out-of-range feature")

        node_gradients = np.asarray(gradients, dtype=float)[indices]
        node_hessians = np.asarray(hessians, dtype=float)[indices]
        total_gradient = float(node_gradients.sum())
        total_hessian = float(node_hessians.sum())
        for feature_idx in features:
            feature_thresholds = self._histogram_thresholds[int(feature_idx)]
            if feature_thresholds.size == 0:
                continue
            bin_codes = self._histogram_bins[indices, feature_idx]
            bin_count = len(feature_thresholds) + 1
            counts = np.bincount(bin_codes, minlength=bin_count)
            gradient_sums = np.bincount(
                bin_codes,
                weights=node_gradients,
                minlength=bin_count,
            )
            hessian_sums = np.bincount(
                bin_codes,
                weights=node_hessians,
                minlength=bin_count,
            )
            cumulative_counts = np.cumsum(counts)
            cumulative_gradients = np.cumsum(gradient_sums)
            cumulative_hessians = np.cumsum(hessian_sums)
            for threshold_idx in range(len(feature_thresholds)):
                n_left = int(cumulative_counts[threshold_idx])
                n_right = len(indices) - n_left
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue
                g_left = float(cumulative_gradients[threshold_idx])
                h_left = float(cumulative_hessians[threshold_idx])
                out.append(
                    {
                        "split_ref": OpaqueSplitReference(
                            state_ref=self._histogram_state_ref,
                            feature_ref=int(feature_idx),
                            bin_ref=int(threshold_idx),
                        ),
                        "g_left": g_left,
                        "h_left": h_left,
                        "g_right": total_gradient - g_left,
                        "h_right": total_hessian - h_left,
                        "n_left": n_left,
                        "n_right": n_right,
                    }
                )
        return out

    def aggregate_local_split_importance(
        self,
        records: list[tuple[OpaqueSplitReference, float]],
    ) -> dict[int, dict[str, float | int]]:
        """Aggregate split usage locally by opaque party feature reference."""
        gains_by_feature: dict[int, list[float]] = {}
        for split_ref, gain in records:
            self._routing_state_for(split_ref)
            feature_ref = split_ref.feature_ref
            gains_by_feature.setdefault(feature_ref, []).append(float(gain))
        output: dict[int, dict[str, float | int]] = {}
        for feature_ref, gains in gains_by_feature.items():
            gain_sum = float(np.sum(gains))
            output[feature_ref] = {
                "split_count": len(gains),
                "gain_sum": gain_sum,
                "gain_mean": gain_sum / len(gains),
            }
        return output

    def _route_with_threshold(
        self, indices: np.ndarray, feature_idx: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature_idx < 0 or feature_idx >= self.n_features:
            raise ValueError("split feature reference is out of range for this party")
        values = self._x[indices, feature_idx]
        left_mask = values <= threshold
        return indices[left_mask], indices[~left_mask]

    def route_split(
        self,
        indices: np.ndarray,
        split_ref: OpaqueSplitReference,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply an opaque training split using only party-owned threshold state."""
        state = self._routing_state_for(split_ref)
        threshold = state._threshold_for(split_ref)
        return self._route_with_threshold(indices, split_ref.feature_ref, threshold)

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
