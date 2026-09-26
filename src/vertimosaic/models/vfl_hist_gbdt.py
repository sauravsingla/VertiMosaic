from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.transport import InMemoryTransport


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TreeNode:
    indices: np.ndarray
    depth: int
    value: float = 0.0
    party: str | None = None
    feature: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class VFLHistGBDT:
    n_estimators: int = 20
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_leaf: int = 10
    min_child_weight: float = 1e-3
    l2_leaf_reg: float = 1.0
    max_bins: int = 16
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    trees_: list[TreeNode] = field(default_factory=list, init=False)
    base_score_: float = field(default=0.0, init=False)
    party_names_: list[str] = field(default_factory=list, init=False)

    @staticmethod
    def _gain(g: float, h: float, reg: float) -> float:
        return (g * g) / (h + reg)

    def _split_gain(self, cand: dict[str, float | int]) -> float:
        gl = float(cand["g_left"])
        hl = float(cand["h_left"])
        gr = float(cand["g_right"])
        hr = float(cand["h_right"])
        if hl < self.min_child_weight or hr < self.min_child_weight:
            return -np.inf
        return 0.5 * (
            self._gain(gl, hl, self.l2_leaf_reg)
            + self._gain(gr, hr, self.l2_leaf_reg)
            - self._gain(gl + gr, hl + hr, self.l2_leaf_reg)
        )

    def _leaf_value(self, gradients: np.ndarray, hessians: np.ndarray, idx: np.ndarray) -> float:
        return -float(gradients[idx].sum()) / (float(hessians[idx].sum()) + self.l2_leaf_reg)

    def _build_node(
        self,
        parties: dict[str, PassiveParty],
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> TreeNode:
        node = TreeNode(indices=indices.copy(), depth=depth)
        node.value = self._leaf_value(gradients, hessians, indices)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        best_gain = 0.0
        best: tuple[PassiveParty, dict[str, float | int]] | None = None
        for party in parties.values():
            candidates = party.candidate_histograms(
                gradients, hessians, indices, self.max_bins, self.min_samples_leaf
            )
            self.transport.send(
                np.empty((len(candidates), 6), dtype=float),
                message_type="candidate_histogram_metadata",
                sender_role=party.name,
                receiver_role="active",
            )
            for cand in candidates:
                gain = self._split_gain(cand)
                if gain > best_gain:
                    best_gain = gain
                    best = (party, cand)
        if best is None:
            return node
        party, cand = best
        feature = int(cand["feature"])
        threshold = float(cand["threshold"])
        left_idx, right_idx = party.route(indices, feature, threshold)
        self.transport.send(
            np.asarray([len(left_idx), len(right_idx)]),
            message_type="partition_routing_counts",
            sender_role=party.name,
            receiver_role="active",
        )
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return node
        node.party = party.name
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_node(parties, gradients, hessians, left_idx, depth + 1)
        node.right = self._build_node(parties, gradients, hessians, right_idx, depth + 1)
        return node

    def _predict_tree(self, tree: TreeNode, parties: dict[str, PassiveParty], n: int) -> np.ndarray:
        out = np.zeros(n, dtype=float)

        def walk(node: TreeNode, idx: np.ndarray) -> None:
            if node.is_leaf:
                out[idx] = node.value
                return
            assert (
                node.party is not None and node.feature is not None and node.threshold is not None
            )
            left_idx, right_idx = parties[node.party].route(idx, node.feature, node.threshold)
            assert node.left is not None and node.right is not None
            walk(node.left, left_idx)
            walk(node.right, right_idx)

        walk(tree, np.arange(n, dtype=int))
        return out

    def fit(self, active: ActiveParty, passive: list[PassiveParty]) -> VFLHistGBDT:
        party_list: list[PassiveParty] = [active, *passive]
        n = active.n_rows
        if any(p.n_rows != n for p in party_list):
            raise ValueError("all VFL parties must align to the same row count")
        parties = {p.name: p for p in party_list}
        self.party_names_ = list(parties)
        y = active.labels
        prevalence = np.clip(y.mean(), 1e-6, 1.0 - 1e-6)
        self.base_score_ = float(np.log(prevalence / (1.0 - prevalence)))
        raw = np.full(n, self.base_score_, dtype=float)
        self.trees_.clear()
        for _ in range(self.n_estimators):
            p = _sigmoid(raw)
            gradients = p - y
            hessians = np.maximum(p * (1.0 - p), 1e-8)
            self.transport.send(
                gradients,
                message_type="gradients",
                sender_role="active",
                receiver_role="passive_parties",
            )
            self.transport.send(
                hessians,
                message_type="hessians",
                sender_role="active",
                receiver_role="passive_parties",
            )
            tree = self._build_node(parties, gradients, hessians, np.arange(n), 0)
            self.trees_.append(tree)
            raw += self.learning_rate * self._predict_tree(tree, parties, n)
        return self

    def decision_function(self, parties: list[PassiveParty]) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("model is not fitted")
        mapping = {p.name: p for p in parties}
        missing = set(self.party_names_) - set(mapping)
        if missing:
            raise ValueError(f"missing parties for inference: {sorted(missing)}")
        n = parties[0].n_rows
        raw = np.full(n, self.base_score_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * self._predict_tree(tree, mapping, n)
        return raw

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        p = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - p, p])
