from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.parties import ActiveParty, HistogramRoutingState, PassiveParty
from vertimosaic.parties.core import HistogramCandidate, OpaqueSplitReference
from vertimosaic.transport import InMemoryTransport, StructuredPayload


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
    split_ref: OpaqueSplitReference | None = None
    gain: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class VFLHistGBDT:
    """CPU vertical histogram gradient boosting research implementation.

    Passive parties receive target-derived gradient/Hessian signals through the
    simulated transport, build histograms from retained local bins, and send only
    aggregate split statistics plus opaque local feature/bin references back through
    ``Message`` objects. The active party sends node membership and selected opaque
    split references through the same transport; the split-owning party performs
    routing locally and returns only aligned partition indices. Numeric thresholds
    stay in immutable party-local routing state learned from training rows.

    This in-process simulator is not cryptographically secure: gradients, Hessians,
    node membership, opaque references, and routing information can leak information.
    See the threat-model documentation.
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
    _routing_states: dict[str, HistogramRoutingState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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

    def _split_gain(self, cand: HistogramCandidate) -> float:
        gl = cand["g_left"]
        hl = cand["h_left"]
        gr = cand["g_right"]
        hr = cand["h_right"]
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

    @staticmethod
    def _candidate_payload(candidates: list[HistogramCandidate]) -> StructuredPayload:
        # Per candidate: 2 opaque integer refs + 4 gradient/Hessian sums + 2 counts.
        scalar_count = len(candidates) * 8
        return StructuredPayload(
            value=candidates,
            shape=(len(candidates), 8),
            scalar_count=scalar_count,
            estimated_bytes=scalar_count * 8,
        )

    @staticmethod
    def _routing_payload(
        left_idx: np.ndarray,
        right_idx: np.ndarray,
    ) -> StructuredPayload:
        scalar_count = int(left_idx.size + right_idx.size)
        return StructuredPayload(
            value=(left_idx, right_idx),
            shape=(scalar_count,),
            scalar_count=scalar_count,
            estimated_bytes=int(left_idx.nbytes + right_idx.nbytes),
        )

    @staticmethod
    def _split_selection_payload(
        split_ref: OpaqueSplitReference,
        indices: np.ndarray,
    ) -> StructuredPayload:
        return StructuredPayload(
            value=(split_ref, indices),
            shape=(len(indices),),
            scalar_count=int(len(indices) + 2),
            estimated_bytes=int(indices.nbytes + 16),
        )

    def _route_selected_split(
        self,
        *,
        party: PassiveParty,
        indices: np.ndarray,
        split_ref: OpaqueSplitReference,
        active_name: str,
        stage: str,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        routing_state = self._routing_states.get(party.name)
        if routing_state is None:
            raise RuntimeError(f"missing party-local routing state for {party.name}")

        route_indices = indices
        route_ref = split_ref
        if party.name != active_name:
            delivered = self.transport.send(
                self._split_selection_payload(split_ref, indices),
                message_type="split_selection",
                sender_role=active_name,
                receiver_role=party.name,
                direction="backward",
                stage=stage,
                step=step,
            )
            route_ref, route_indices = delivered
            route_indices = np.asarray(route_indices, dtype=int)

        left_idx, right_idx = party.route_split(route_indices, route_ref, routing_state)
        if party.name != active_name:
            delivered_routing = self.transport.send(
                self._routing_payload(left_idx, right_idx),
                message_type="partition_routing_indices",
                sender_role=party.name,
                receiver_role=active_name,
                direction="forward",
                stage=stage,
                step=step,
            )
            left_idx, right_idx = delivered_routing
        return np.asarray(left_idx, dtype=int), np.asarray(right_idx, dtype=int)

    def _build_node(
        self,
        parties: dict[str, PassiveParty],
        gradients: np.ndarray,
        hessians: np.ndarray,
        gradient_signals: dict[str, np.ndarray],
        hessian_signals: dict[str, np.ndarray],
        indices: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        leaf_count: list[int],
        tree_index: int,
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

        active_name = next(iter(parties))
        best_gain = 0.0
        best: tuple[PassiveParty, HistogramCandidate] | None = None
        for party in parties.values():
            feature_indices = self._feature_indices(party, rng)
            party_indices = indices
            party_feature_indices = feature_indices
            if party.name != active_name:
                party_indices = np.asarray(
                    self.transport.send(
                        indices,
                        message_type="node_membership",
                        sender_role=active_name,
                        receiver_role=party.name,
                        direction="backward",
                        stage="tree",
                        step=tree_index,
                    ),
                    dtype=int,
                )
                party_feature_indices = np.asarray(
                    self.transport.send(
                        feature_indices,
                        message_type="feature_subsample_refs",
                        sender_role=active_name,
                        receiver_role=party.name,
                        direction="backward",
                        stage="tree",
                        step=tree_index,
                    ),
                    dtype=int,
                )

            local_candidates = party.candidate_histograms(
                gradient_signals[party.name],
                hessian_signals[party.name],
                party_indices,
                self.max_bins,
                self.min_samples_leaf,
                party_feature_indices,
            )
            candidates = local_candidates
            if party.name != active_name:
                candidates = self.transport.send(
                    self._candidate_payload(local_candidates),
                    message_type="candidate_histogram_statistics",
                    sender_role=party.name,
                    receiver_role=active_name,
                    direction="forward",
                    stage="tree",
                    step=tree_index,
                )

            for cand in candidates:
                gain = self._split_gain(cand)
                if gain > best_gain:
                    best_gain = gain
                    best = (party, cand)

        if best is None:
            return node

        party, cand = best
        split_ref = cand["split_ref"]
        left_idx, right_idx = self._route_selected_split(
            party=party,
            indices=indices,
            split_ref=split_ref,
            active_name=active_name,
            stage="tree",
            step=tree_index,
        )
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return node

        node.party = party.name
        node.split_ref = split_ref
        node.gain = float(best_gain)
        leaf_count[0] += 1
        node.left = self._build_node(
            parties,
            gradients,
            hessians,
            gradient_signals,
            hessian_signals,
            left_idx,
            depth + 1,
            rng,
            leaf_count,
            tree_index,
        )
        node.right = self._build_node(
            parties,
            gradients,
            hessians,
            gradient_signals,
            hessian_signals,
            right_idx,
            depth + 1,
            rng,
            leaf_count,
            tree_index,
        )
        return node

    def _predict_tree(
        self,
        tree: TreeNode,
        parties: dict[str, PassiveParty],
        n_rows: int,
        *,
        tree_index: int,
        stage: str,
    ) -> np.ndarray:
        out = np.zeros(n_rows, dtype=float)
        active_name = self.party_names_[0]

        def walk(node: TreeNode, idx: np.ndarray) -> None:
            if node.is_leaf:
                out[idx] = node.value
                return
            if node.party is None or node.split_ref is None:
                raise RuntimeError("non-leaf node is missing split metadata")
            if node.party not in parties:
                if self.missing_party_policy == "zero_contribution":
                    out[idx] = 0.0
                    return
                raise ValueError(f"missing split-owning party for inference: {node.party}")
            left_idx, right_idx = self._route_selected_split(
                party=parties[node.party],
                indices=idx,
                split_ref=node.split_ref,
                active_name=active_name,
                stage=stage,
                step=tree_index,
            )
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
        for party in party_list:
            party.prepare_histogram_bins(self.max_bins)
        self._routing_states = {
            party.name: party.export_histogram_routing_state() for party in party_list
        }

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
        validation_labels: np.ndarray | None = None
        if validation_active is not None:
            validation_passive = validation_passive or []
            validation_mapping = self._party_mapping(validation_active, validation_passive)
            if set(validation_mapping) != set(parties):
                raise ValueError("validation data must provide the same VFL parties as training")
            validation_n = validation_active.n_rows
            if any(party.n_rows != validation_n for party in validation_mapping.values()):
                raise ValueError("validation parties must align to the same row count")
            validation_raw = np.full(validation_n, self.base_score_, dtype=float)
            validation_labels = validation_active.labels

        best_loss = np.inf
        rounds_without_improvement = 0
        best_tree_count = 0
        for tree_index in range(self.n_estimators):
            probability = _sigmoid(raw)
            gradients = probability - y
            hessians = np.maximum(probability * (1.0 - probability), 1e-8)
            gradient_signals = {active.name: gradients}
            hessian_signals = {active.name: hessians}
            for party in passive:
                gradient_signals[party.name] = np.asarray(
                    self.transport.send(
                        gradients,
                        message_type="gradients",
                        sender_role=active.name,
                        receiver_role=party.name,
                        direction="backward",
                        stage="tree",
                        step=tree_index,
                    ),
                    dtype=float,
                )
                hessian_signals[party.name] = np.asarray(
                    self.transport.send(
                        hessians,
                        message_type="hessians",
                        sender_role=active.name,
                        receiver_role=party.name,
                        direction="backward",
                        stage="tree",
                        step=tree_index,
                    ),
                    dtype=float,
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
                gradient_signals,
                hessian_signals,
                tree_indices,
                0,
                rng,
                [1],
                tree_index,
            )
            self.trees_.append(tree)
            raw += self.learning_rate * self._predict_tree(
                tree,
                parties,
                n,
                tree_index=tree_index,
                stage="training_routing",
            )
            self.training_loss_history_.append(_binary_log_loss(y, _sigmoid(raw)))

            if (
                validation_mapping is not None
                and validation_raw is not None
                and validation_labels is not None
            ):
                validation_raw += self.learning_rate * self._predict_tree(
                    tree,
                    validation_mapping,
                    len(validation_raw),
                    tree_index=tree_index,
                    stage="validation_routing",
                )
                validation_loss = _binary_log_loss(
                    validation_labels,
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
        for tree_index, tree in enumerate(self.trees_):
            raw += self.learning_rate * self._predict_tree(
                tree,
                mapping,
                n,
                tree_index=tree_index,
                stage="inference_routing",
            )
        return raw

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        probability = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - probability, probability])
