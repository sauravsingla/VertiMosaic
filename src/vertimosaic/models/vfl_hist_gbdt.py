from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.transport import InMemoryTransport


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    eps = 1e-12
    probability = np.clip(probability, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)))


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
    """CPU vertical histogram gradient boosting research implementation.

    Passive parties expose only aggregate split statistics plus opaque feature/bin
    references to the active-party protocol. This provides data locality and feature
    separation, not cryptographic confidentiality.
    """

    n_estimators: int = 20
    learning_rate: float = 0.1
    max_depth: int = 3
    max_leaves: int | None = None
    min_samples_leaf: int = 10
    min_child_weight: float = 1e-3
    l2_leaf_reg: float = 1.0
    max_bins: int = 16
    subsample: float = 1.0
    feature_subsample: float = 1.0
    early_stopping_rounds: int | None = None
    missing_party_policy: str = "error"
    seed: int = 42
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    trees_: list[TreeNode] = field(default_factory=list, init=False)
    base_score_: float = field(default=0.0, init=False)
    party_names_: list[str] = field(default_factory=list, init=False)
    training_loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    best_iteration_: int | None = field(default=None, init=False)

    def _validate_hyperparameters(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_leaves is not None and self.max_leaves < 2:
            raise ValueError("max_leaves must be at least 2 when supplied")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.min_child_weight < 0 or self.l2_leaf_reg < 0:
            raise ValueError("child and leaf regularization values must be non-negative")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.feature_subsample <= 1.0:
            raise ValueError("feature_subsample must be in (0, 1]")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive when supplied")
        if self.missing_party_policy not in {"error", "zero_contribution"}:
            raise ValueError("missing_party_policy must be error or zero_contribution")

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

    def _feature_indices(self, party: PassiveParty, rng: np.random.Generator) -> np.ndarray:
        if party.n_features == 0:
            return np.empty(0, dtype=int)
        if self.feature_subsample >= 1.0:
            return np.arange(party.n_features, dtype=int)
        count = max(1, int(np.ceil(party.n_features * self.feature_subsample)))
        return np.sort(rng.choice(party.n_features, size=count, replace=False)).astype(int)

    def _build_node(
        self,
        parties: dict[str, PassiveParty],
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        leaf_count: list[int],
    ) -> TreeNode:
        node = TreeNode(indices=indices.copy(), depth=depth)
        node.value = self._leaf_value(gradients, hessians, indices)
        leaf_limit_reached = self.max_leaves is not None and leaf_count[0] >= self.max_leaves
        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
            or leaf_limit_reached
        ):
            return node
        best_gain = 0.0
        best: tuple[PassiveParty, dict[str, float | int]] | None = None
        for party in parties.values():
            feature_indices = self._feature_indices(party, rng)
            candidates = party.candidate_histograms(
                gradients,
                hessians,
                indices,
                self.max_bins,
                self.min_samples_leaf,
                feature_indices,
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
        leaf_count[0] += 1
        node.left = self._build_node(
            parties, gradients, hessians, left_idx, depth + 1, rng, leaf_count
        )
        node.right = self._build_node(
            parties, gradients, hessians, right_idx, depth + 1, rng, leaf_count
        )
        return node

    def _predict_tree(
        self, tree: TreeNode, parties: dict[str, PassiveParty], n_rows: int
    ) -> np.ndarray:
        out = np.zeros(n_rows, dtype=float)

        def walk(node: TreeNode, idx: np.ndarray) -> None:
            if node.is_leaf:
                out[idx] = node.value
                return
            if node.party is None or node.feature is None or node.threshold is None:
                raise RuntimeError("non-leaf node is missing split metadata")
            if node.party not in parties:
                if self.missing_party_policy == "zero_contribution":
                    out[idx] = 0.0
                    return
                raise ValueError(f"missing split-owning party for inference: {node.party}")
            left_idx, right_idx = parties[node.party].route(idx, node.feature, node.threshold)
            if node.left is None or node.right is None:
                raise RuntimeError("non-leaf node is missing child nodes")
            walk(node.left, left_idx)
            walk(node.right, right_idx)

        walk(tree, np.arange(n_rows, dtype=int))
        return out

    @staticmethod
    def _party_mapping(active: ActiveParty, passive: list[PassiveParty]) -> dict[str, PassiveParty]:
        return {party.name: party for party in [active, *passive]}

    def fit(
        self,
        active: ActiveParty,
        passive: list[PassiveParty],
        validation_active: ActiveParty | None = None,
        validation_passive: list[PassiveParty] | None = None,
    ) -> VFLHistGBDT:
        self._validate_hyperparameters()
        party_list: list[PassiveParty] = [active, *passive]
        n = active.n_rows
        if any(party.n_rows != n for party in party_list):
            raise ValueError("all VFL parties must align to the same row count")
        if self.early_stopping_rounds is not None and validation_active is None:
            raise ValueError("validation data are required when early stopping is enabled")
        parties = {party.name: party for party in party_list}
        self.party_names_ = list(parties)
        y = active.labels
        prevalence = np.clip(y.mean(), 1e-6, 1.0 - 1e-6)
        self.base_score_ = float(np.log(prevalence / (1.0 - prevalence)))
        raw = np.full(n, self.base_score_, dtype=float)
        self.trees_.clear()
        self.training_loss_history_.clear()
        self.validation_loss_history_.clear()
        self.best_iteration_ = None
        rng = np.random.default_rng(self.seed)

        validation_mapping: dict[str, PassiveParty] | None = None
        validation_raw: np.ndarray | None = None
        if validation_active is not None:
            validation_passive = validation_passive or []
            validation_mapping = self._party_mapping(validation_active, validation_passive)
            if set(validation_mapping) != set(parties):
                raise ValueError("validation data must provide the same VFL parties as training")
            validation_n = validation_active.n_rows
            if any(party.n_rows != validation_n for party in validation_mapping.values()):
                raise ValueError("validation parties must align to the same row count")
            validation_raw = np.full(validation_n, self.base_score_, dtype=float)

        best_loss = np.inf
        rounds_without_improvement = 0
        best_tree_count = 0
        for _ in range(self.n_estimators):
            probability = _sigmoid(raw)
            gradients = probability - y
            hessians = np.maximum(probability * (1.0 - probability), 1e-8)
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
            if self.subsample >= 1.0:
                tree_indices = np.arange(n, dtype=int)
            else:
                sample_size = max(
                    2 * self.min_samples_leaf,
                    int(np.ceil(n * self.subsample)),
                )
                sample_size = min(sample_size, n)
                tree_indices = np.sort(rng.choice(n, size=sample_size, replace=False)).astype(int)
            tree = self._build_node(
                parties,
                gradients,
                hessians,
                tree_indices,
                0,
                rng,
                [1],
            )
            self.trees_.append(tree)
            raw += self.learning_rate * self._predict_tree(tree, parties, n)
            self.training_loss_history_.append(_binary_log_loss(y, _sigmoid(raw)))

            if validation_mapping is not None and validation_raw is not None:
                validation_raw += self.learning_rate * self._predict_tree(
                    tree, validation_mapping, len(validation_raw)
                )
                validation_loss = _binary_log_loss(
                    validation_active.labels,
                    _sigmoid(validation_raw),
                )
                self.validation_loss_history_.append(validation_loss)
                if validation_loss < best_loss - 1e-12:
                    best_loss = validation_loss
                    best_tree_count = len(self.trees_)
                    self.best_iteration_ = best_tree_count - 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    break

        if self.early_stopping_rounds is not None and best_tree_count:
            self.trees_ = self.trees_[:best_tree_count]
        elif self.trees_:
            self.best_iteration_ = len(self.trees_) - 1
        return self

    def decision_function(self, parties: list[PassiveParty]) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("model is not fitted")
        if not parties:
            raise ValueError("at least one party is required")
        mapping = {party.name: party for party in parties}
        missing = set(self.party_names_) - set(mapping)
        if missing and self.missing_party_policy == "error":
            raise ValueError(f"missing parties for inference: {sorted(missing)}")
        n = parties[0].n_rows
        if any(party.n_rows != n for party in parties):
            raise ValueError("inference parties must have equal row counts")
        raw = np.full(n, self.base_score_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * self._predict_tree(tree, mapping, n)
        return raw

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        probability = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - probability, probability])
