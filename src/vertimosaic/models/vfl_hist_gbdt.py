"""Research vertical histogram gradient-boosted decision trees.

The implementation is intentionally compact and auditable. Parties retain binned feature
matrices locally and exchange aggregated gradient/Hessian split statistics plus routing masks.
It provides feature locality, not cryptographic confidentiality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from vertimosaic.parties import ActiveParty, Party
from vertimosaic.transport import InMemoryTransport, Message


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


class SplitCandidate(NamedTuple):
    gain: float
    party: str
    feature: int
    bin_id: int


@dataclass(slots=True)
class TreeNode:
    value: float = 0.0
    party: str | None = None
    feature: int | None = None
    bin_id: int | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(slots=True)
class _BinnedParty:
    name: str
    bins: np.ndarray
    edges: list[np.ndarray]


@dataclass(slots=True)
class VFLHistGBDT:
    """CPU-oriented vertically partitioned histogram GBDT for binary classification."""

    n_estimators: int = 30
    learning_rate: float = 0.1
    max_depth: int = 3
    max_bins: int = 32
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    l2: float = 1.0
    min_gain: float = 1e-10
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    trees_: list[TreeNode] = field(default_factory=list, init=False)
    base_score_: float = field(default=0.0, init=False)
    party_order_: tuple[str, ...] = field(default_factory=tuple, init=False)
    edges_: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    feature_importance_: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def fit(self, parties: list[Party], active_party: ActiveParty) -> "VFLHistGBDT":
        if not parties:
            raise ValueError("at least one party is required")
        n = active_party.n_samples
        if any(p.n_samples != n for p in parties):
            raise ValueError("all parties must be aligned")
        self.party_order_ = tuple(p.name for p in parties)
        binned = [self._bin_party(p, fit=True) for p in parties]
        self.feature_importance_ = {p.name: np.zeros(p.n_features) for p in parties}
        prevalence = float(np.clip(np.mean(active_party.y), 1e-6, 1.0 - 1e-6))
        self.base_score_ = float(np.log(prevalence / (1.0 - prevalence)))
        raw_pred = np.full(n, self.base_score_, dtype=float)
        self.trees_.clear()

        for _ in range(self.n_estimators):
            prob = _sigmoid(raw_pred)
            grad = prob - active_party.y
            hess = np.maximum(prob * (1.0 - prob), 1e-12)
            root = self._build_node(binned, grad, hess, np.arange(n), depth=0)
            update = self._predict_tree(root, binned)
            raw_pred += self.learning_rate * update
            self.trees_.append(root)
        return self

    def predict_proba(self, parties: list[Party]) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("model is not fitted")
        if tuple(p.name for p in parties) != self.party_order_:
            raise ValueError("prediction parties must match fitted party order")
        binned = [self._bin_party(p, fit=False) for p in parties]
        raw = np.full(parties[0].n_samples, self.base_score_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * self._predict_tree(tree, binned)
        p = _sigmoid(raw)
        return np.column_stack([1.0 - p, p])

    def predict(self, parties: list[Party], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(parties)[:, 1] >= threshold).astype(int)

    def _bin_party(self, party: Party, *, fit: bool) -> _BinnedParty:
        bins = np.zeros_like(party.X, dtype=np.int16)
        edges: list[np.ndarray] = []
        for j in range(party.n_features):
            col = party.X[:, j]
            if fit:
                qs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
                edge = np.unique(np.quantile(col, qs))
            else:
                edge = self.edges_[party.name][j]
            bins[:, j] = np.searchsorted(edge, col, side="right")
            edges.append(edge)
        if fit:
            self.edges_[party.name] = edges
        return _BinnedParty(party.name, bins, edges)

    def _build_node(
        self,
        parties: list[_BinnedParty],
        grad: np.ndarray,
        hess: np.ndarray,
        idx: np.ndarray,
        *,
        depth: int,
    ) -> TreeNode:
        g_total = float(np.sum(grad[idx]))
        h_total = float(np.sum(hess[idx]))
        leaf_value = -g_total / (h_total + self.l2)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return TreeNode(value=leaf_value)

        candidate = self._best_split(parties, grad, hess, idx)
        if candidate is None or candidate.gain <= self.min_gain:
            return TreeNode(value=leaf_value)
        owner = next(p for p in parties if p.name == candidate.party)
        mask = owner.bins[idx, candidate.feature] <= candidate.bin_id
        left_idx, right_idx = idx[mask], idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return TreeNode(value=leaf_value)
        route = self.transport.send(
            Message("routing_mask", candidate.party, "coordinator", {"mask": mask.astype(np.int8)})
        ).payload["mask"]
        left_idx, right_idx = idx[route.astype(bool)], idx[~route.astype(bool)]
        self.feature_importance_[candidate.party][candidate.feature] += candidate.gain
        return TreeNode(
            value=leaf_value,
            party=candidate.party,
            feature=candidate.feature,
            bin_id=candidate.bin_id,
            left=self._build_node(parties, grad, hess, left_idx, depth=depth + 1),
            right=self._build_node(parties, grad, hess, right_idx, depth=depth + 1),
        )

    def _best_split(
        self,
        parties: list[_BinnedParty],
        grad: np.ndarray,
        hess: np.ndarray,
        idx: np.ndarray,
    ) -> SplitCandidate | None:
        best: SplitCandidate | None = None
        g_total = float(np.sum(grad[idx]))
        h_total = float(np.sum(hess[idx]))
        parent_score = g_total * g_total / (h_total + self.l2)
        for party in parties:
            for feature in range(party.bins.shape[1]):
                values = party.bins[idx, feature]
                n_bins = int(values.max()) + 1 if len(values) else 0
                if n_bins <= 1:
                    continue
                g_hist = np.bincount(values, weights=grad[idx], minlength=n_bins)
                h_hist = np.bincount(values, weights=hess[idx], minlength=n_bins)
                n_hist = np.bincount(values, minlength=n_bins)
                summary = np.stack([g_hist, h_hist, n_hist], axis=1)
                received = self.transport.send(
                    Message(
                        "split_histogram",
                        party.name,
                        "coordinator",
                        {"summary": summary, "feature_ref": feature},
                    )
                ).payload["summary"]
                g_left = np.cumsum(received[:, 0])[:-1]
                h_left = np.cumsum(received[:, 1])[:-1]
                n_left = np.cumsum(received[:, 2])[:-1]
                g_right = g_total - g_left
                h_right = h_total - h_left
                n_right = len(idx) - n_left
                valid = (
                    (n_left >= self.min_samples_leaf)
                    & (n_right >= self.min_samples_leaf)
                    & (h_left >= self.min_child_weight)
                    & (h_right >= self.min_child_weight)
                )
                gains = np.full_like(g_left, -np.inf, dtype=float)
                gains[valid] = 0.5 * (
                    g_left[valid] ** 2 / (h_left[valid] + self.l2)
                    + g_right[valid] ** 2 / (h_right[valid] + self.l2)
                    - parent_score
                )
                if gains.size == 0:
                    continue
                bin_id = int(np.argmax(gains))
                gain = float(gains[bin_id])
                cand = SplitCandidate(gain, party.name, feature, bin_id)
                if best is None or cand.gain > best.gain:
                    best = cand
        return best

    def _predict_tree(self, root: TreeNode, parties: list[_BinnedParty]) -> np.ndarray:
        by_name = {p.name: p for p in parties}
        n = parties[0].bins.shape[0]
        out = np.empty(n, dtype=float)

        def walk(node: TreeNode, idx: np.ndarray) -> None:
            if node.is_leaf:
                out[idx] = node.value
                return
            assert node.party is not None and node.feature is not None and node.bin_id is not None
            owner = by_name[node.party]
            mask = owner.bins[idx, node.feature] <= node.bin_id
            assert node.left is not None and node.right is not None
            walk(node.left, idx[mask])
            walk(node.right, idx[~mask])

        walk(root, np.arange(n))
        return out
