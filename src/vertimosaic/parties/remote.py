# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vertimosaic.parties.core import (
    HistogramCandidate,
    HistogramRoutingState,
    OpaqueSplitReference,
    PassiveParty,
)
from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport

_METADATA = "party.metadata"
_LOCAL_LOGITS = "party.local_logits"
_LOCAL_GRADIENT = "party.local_gradient"
_HIST_PREPARE = "party.histogram.prepare"
_HIST_EXPORT = "party.histogram.export_state"
_HIST_SIGNALS = "party.histogram.set_signals"
_HIST_CLEAR_SIGNALS = "party.histogram.clear_signals"
_HIST_CANDIDATES = "party.histogram.candidates"
_HIST_ROUTE = "party.histogram.route"
_HIST_SHARE = "party.histogram.share_state"
_HIST_IMPORTANCE = "party.histogram.aggregate_importance"


@dataclass
class RemotePartyService:
    """Expose party-local VFL computations without moving raw feature matrices.

    One service can own multiple row partitions for the same organization, typically
    ``train`` and ``validation``. Histogram thresholds, binned matrices, gradient/
    Hessian round state, and routing decisions stay inside this service boundary.
    The coordinator receives only metadata, aggregate split statistics, opaque split
    references, opaque routing handles, and routed row indices.
    """

    party: PassiveParty
    allowed_senders: set[str] | None = None
    partitions: dict[str, PassiveParty] = field(default_factory=dict)
    _signal_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        normalized = dict(self.partitions)
        normalized.setdefault("train", self.party)
        for partition, candidate in normalized.items():
            if not partition:
                raise ValueError("remote party partition names must be non-empty")
            if candidate.name != self.party.name:
                raise ValueError("all remote service partitions must belong to the same party")
            if candidate.n_features != self.party.n_features:
                raise ValueError("all remote service partitions must have equal feature width")
        self.partitions = normalized

    def _party_for(self, payload: Any) -> tuple[str, PassiveParty]:
        partition = "train"
        if isinstance(payload, dict) and "partition" in payload:
            partition = str(payload["partition"])
        party = self.partitions.get(partition)
        if party is None:
            raise ValueError(f"unknown remote party partition: {partition}")
        return partition, party

    @staticmethod
    def _mapping(payload: Any, operation: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"{operation} expects a mapping payload")
        return payload

    @staticmethod
    def _pair(payload: Any, operation: str) -> tuple[Any, Any]:
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise ValueError(f"{operation} expects a two-item tuple payload")
        return payload[0], payload[1]

    @staticmethod
    def _indices(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=int).reshape(-1)

    def handle(self, message_type: str, sender_role: str, payload: Any) -> Any:
        if self.allowed_senders is not None and sender_role not in self.allowed_senders:
            raise ValueError(f"sender role {sender_role!r} is not authorized for this party")
        partition, party = self._party_for(payload)

        if message_type == _METADATA:
            return {
                "name": party.name,
                "partition": partition,
                "n_rows": party.n_rows,
                "n_features": party.n_features,
            }
        if message_type == _LOCAL_LOGITS:
            if isinstance(payload, dict):
                request = self._mapping(payload, _LOCAL_LOGITS)
                weights = request.get("weights")
                indices = request.get("indices")
            else:
                weights, indices = self._pair(payload, _LOCAL_LOGITS)
            return party.local_logits(
                np.asarray(weights, dtype=float),
                self._indices(indices),
            )
        if message_type == _LOCAL_GRADIENT:
            if isinstance(payload, dict):
                request = self._mapping(payload, _LOCAL_GRADIENT)
                residual = request.get("residual")
                indices = request.get("indices")
            else:
                residual, indices = self._pair(payload, _LOCAL_GRADIENT)
            return party.local_gradient(
                np.asarray(residual, dtype=float),
                self._indices(indices),
            )
        if message_type == _HIST_PREPARE:
            request = self._mapping(payload, _HIST_PREPARE)
            max_bins = int(request["max_bins"])
            party.prepare_histogram_bins(max_bins)
            return party.export_histogram_routing_state()
        if message_type == _HIST_EXPORT:
            return party.export_histogram_routing_state()
        if message_type == _HIST_SIGNALS:
            request = self._mapping(payload, _HIST_SIGNALS)
            signal_ref = str(request["signal_ref"])
            gradients = np.asarray(request["gradients"], dtype=float).reshape(-1)
            hessians = np.asarray(request["hessians"], dtype=float).reshape(-1)
            if gradients.shape != hessians.shape or gradients.size != party.n_rows:
                raise ValueError("gradient/Hessian signals must match the remote partition rows")
            if not np.isfinite(gradients).all() or not np.isfinite(hessians).all():
                raise ValueError("gradient/Hessian signals must be finite")
            for key in [key for key in self._signal_cache if key[0] == partition]:
                self._signal_cache.pop(key, None)
            self._signal_cache[(partition, signal_ref)] = (
                gradients.copy(),
                hessians.copy(),
            )
            return {"signal_ref": signal_ref, "rows": int(gradients.size)}
        if message_type == _HIST_CLEAR_SIGNALS:
            request = self._mapping(payload, _HIST_CLEAR_SIGNALS)
            signal_ref = request.get("signal_ref")
            if signal_ref is None:
                keys = [key for key in self._signal_cache if key[0] == partition]
                for key in keys:
                    self._signal_cache.pop(key, None)
            else:
                self._signal_cache.pop((partition, str(signal_ref)), None)
            return None
        if message_type == _HIST_CANDIDATES:
            request = self._mapping(payload, _HIST_CANDIDATES)
            signal_ref = request.get("signal_ref")
            if signal_ref is not None:
                cached = self._signal_cache.get((partition, str(signal_ref)))
                if cached is None:
                    raise ValueError("unknown or expired gradient/Hessian signal reference")
                gradients, hessians = cached
            else:
                gradients = np.asarray(request["gradients"], dtype=float).reshape(-1)
                hessians = np.asarray(request["hessians"], dtype=float).reshape(-1)
            feature_indices = request.get("feature_indices")
            return party.candidate_histograms(
                gradients,
                hessians,
                np.asarray(request["indices"], dtype=int).reshape(-1),
                int(request["max_bins"]),
                int(request["min_samples_leaf"]),
                None
                if feature_indices is None
                else np.asarray(feature_indices, dtype=int).reshape(-1),
            )
        if message_type == _HIST_ROUTE:
            request = self._mapping(payload, _HIST_ROUTE)
            split_ref = request.get("split_ref")
            routing_state = request.get("routing_state")
            if not isinstance(split_ref, OpaqueSplitReference):
                raise ValueError("histogram routing requires an opaque split reference")
            if routing_state is not None and not isinstance(routing_state, HistogramRoutingState):
                raise ValueError("histogram routing state must be an opaque routing handle")
            return party.route_split(
                np.asarray(request["indices"], dtype=int).reshape(-1),
                split_ref,
                routing_state,
            )
        if message_type == _HIST_SHARE:
            request = self._mapping(payload, _HIST_SHARE)
            target_partition = str(request["target_partition"])
            target = self.partitions.get(target_partition)
            if target is None:
                raise ValueError(f"unknown target remote party partition: {target_partition}")
            party.share_histogram_routing_state_with(target)
            return target.export_histogram_routing_state()
        if message_type == _HIST_IMPORTANCE:
            request = self._mapping(payload, _HIST_IMPORTANCE)
            records_raw = request.get("records")
            if not isinstance(records_raw, list):
                raise ValueError("histogram importance records must be a list")
            records: list[tuple[OpaqueSplitReference, float]] = []
            for item in records_raw:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("histogram importance records must be two-item tuples")
                split_ref, gain = item
                if not isinstance(split_ref, OpaqueSplitReference):
                    raise ValueError("histogram importance requires opaque split references")
                records.append((split_ref, float(gain)))
            return party.aggregate_local_split_importance(records)
        raise ValueError(f"unsupported remote party operation: {message_type}")

    def make_server(
        self,
        address: tuple[str, int],
        *,
        bearer_token: str | None = None,
        path: str = "/v1/messages",
    ) -> ReferenceRelayServer:
        return ReferenceRelayServer(
            address,
            receiver_role=self.party.name,
            bearer_token=bearer_token,
            path=path,
            request_handler=self.handle,
        )


@dataclass
class RemotePassiveParty:
    """Metadata-only proxy for a passive party whose raw features are remote."""

    name: str
    n_rows: int
    n_features: int
    transport: RemoteHTTPTransport
    coordinator_role: str = "bank"
    partition: str = "train"
    _histogram_routing_state: HistogramRoutingState | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_signal_ref: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def discover(
        cls,
        name: str,
        transport: RemoteHTTPTransport,
        *,
        coordinator_role: str = "bank",
        partition: str = "train",
    ) -> RemotePassiveParty:
        metadata = transport.send(
            {"partition": partition},
            message_type=_METADATA,
            sender_role=coordinator_role,
            receiver_role=name,
            direction="rpc",
            stage="discovery",
        )
        if not isinstance(metadata, dict):
            raise RuntimeError("remote party metadata response must be a mapping")
        if metadata.get("name") != name:
            raise RuntimeError("remote party metadata name does not match requested role")
        if metadata.get("partition") != partition:
            raise RuntimeError("remote party metadata partition does not match request")
        return cls(
            name=name,
            n_rows=int(metadata["n_rows"]),
            n_features=int(metadata["n_features"]),
            transport=transport,
            coordinator_role=coordinator_role,
            partition=partition,
        )

    def _rpc(self, message_type: str, payload: dict[str, Any], *, stage: str) -> Any:
        request = {"partition": self.partition, **payload}
        return self.transport.send(
            request,
            message_type=message_type,
            sender_role=self.coordinator_role,
            receiver_role=self.name,
            direction="rpc",
            stage=stage,
        )

    @property
    def histogram_bins_ready(self) -> bool:
        return self._histogram_routing_state is not None

    def local_logits(
        self,
        weights: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        delivered = self._rpc(
            _LOCAL_LOGITS,
            {"weights": np.asarray(weights, dtype=float), "indices": indices},
            stage="party_compute",
        )
        return np.asarray(delivered, dtype=float)

    def local_gradient(
        self,
        residual: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        delivered = self._rpc(
            _LOCAL_GRADIENT,
            {"residual": np.asarray(residual, dtype=float), "indices": indices},
            stage="party_compute",
        )
        return np.asarray(delivered, dtype=float)

    def prepare_histogram_bins(self, max_bins: int) -> None:
        state = self._rpc(
            _HIST_PREPARE,
            {"max_bins": int(max_bins)},
            stage="histogram_prepare",
        )
        if not isinstance(state, HistogramRoutingState):
            raise RuntimeError("remote histogram prepare did not return a routing handle")
        self._histogram_routing_state = state

    def export_histogram_routing_state(self) -> HistogramRoutingState:
        if self._histogram_routing_state is None:
            state = self._rpc(_HIST_EXPORT, {}, stage="histogram_state")
            if not isinstance(state, HistogramRoutingState):
                raise RuntimeError("remote histogram state response must be an opaque handle")
            self._histogram_routing_state = state
        return self._histogram_routing_state

    def share_histogram_routing_state_with(self, other: Any) -> None:
        if not isinstance(other, RemotePassiveParty):
            raise ValueError("remote routing state can only be shared with a remote partition")
        if other.name != self.name or other.n_features != self.n_features:
            raise ValueError(
                "remote routing state sharing requires the same party and feature width"
            )
        source_endpoint = self.transport.endpoints.get(self.name)
        target_endpoint = other.transport.endpoints.get(other.name)
        if source_endpoint != target_endpoint:
            raise ValueError("remote routing state sharing requires partitions on the same service")
        state = self._rpc(
            _HIST_SHARE,
            {"target_partition": other.partition},
            stage="histogram_state",
        )
        if not isinstance(state, HistogramRoutingState):
            raise RuntimeError("remote routing state sharing did not return an opaque handle")
        other._histogram_routing_state = state

    def set_gradient_hessian(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        *,
        signal_ref: str,
    ) -> None:
        response = self._rpc(
            _HIST_SIGNALS,
            {
                "signal_ref": signal_ref,
                "gradients": np.asarray(gradients, dtype=float),
                "hessians": np.asarray(hessians, dtype=float),
            },
            stage="histogram_signals",
        )
        if not isinstance(response, dict) or response.get("signal_ref") != signal_ref:
            raise RuntimeError("remote gradient/Hessian signal registration failed")
        self._active_signal_ref = signal_ref

    def clear_gradient_hessian(self, signal_ref: str | None = None) -> None:
        ref = signal_ref if signal_ref is not None else self._active_signal_ref
        self._rpc(
            _HIST_CLEAR_SIGNALS,
            {"signal_ref": ref},
            stage="histogram_signals",
        )
        if ref == self._active_signal_ref:
            self._active_signal_ref = None

    @staticmethod
    def _signal_reference(gradients: np.ndarray, hessians: np.ndarray) -> str:
        digest = hashlib.sha256()
        gradient_values = np.ascontiguousarray(gradients, dtype=float)
        hessian_values = np.ascontiguousarray(hessians, dtype=float)
        digest.update(gradient_values.shape[0].to_bytes(8, "big", signed=False))
        digest.update(gradient_values.tobytes())
        digest.update(hessian_values.tobytes())
        return f"round-{digest.hexdigest()[:24]}"

    def candidate_histograms(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        max_bins: int,
        min_samples_leaf: int,
        feature_indices: np.ndarray | None = None,
    ) -> list[HistogramCandidate]:
        gradient_values = np.asarray(gradients, dtype=float).reshape(-1)
        hessian_values = np.asarray(hessians, dtype=float).reshape(-1)
        signal_ref = self._signal_reference(gradient_values, hessian_values)
        if self._active_signal_ref != signal_ref:
            self.set_gradient_hessian(
                gradient_values,
                hessian_values,
                signal_ref=signal_ref,
            )
        delivered = self._rpc(
            _HIST_CANDIDATES,
            {
                "signal_ref": signal_ref,
                "indices": np.asarray(indices, dtype=int),
                "max_bins": int(max_bins),
                "min_samples_leaf": int(min_samples_leaf),
                "feature_indices": feature_indices,
            },
            stage="histogram_candidates",
        )
        if not isinstance(delivered, list):
            raise RuntimeError("remote histogram candidate response must be a list")
        return delivered

    def route_split(
        self,
        indices: np.ndarray,
        split_ref: OpaqueSplitReference,
        routing_state: HistogramRoutingState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = routing_state or self.export_histogram_routing_state()
        delivered = self._rpc(
            _HIST_ROUTE,
            {
                "indices": np.asarray(indices, dtype=int),
                "split_ref": split_ref,
                "routing_state": state,
            },
            stage="histogram_routing",
        )
        if not isinstance(delivered, tuple) or len(delivered) != 2:
            raise RuntimeError("remote histogram routing response must contain two partitions")
        left, right = delivered
        return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

    def aggregate_local_split_importance(
        self,
        records: list[tuple[OpaqueSplitReference, float]],
    ) -> dict[int, dict[str, float | int]]:
        delivered = self._rpc(
            _HIST_IMPORTANCE,
            {"records": records},
            stage="histogram_importance",
        )
        if not isinstance(delivered, dict):
            raise RuntimeError("remote histogram importance response must be a mapping")
        return {int(key): value for key, value in delivered.items()}
