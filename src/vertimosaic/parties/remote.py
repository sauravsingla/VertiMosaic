# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vertimosaic.parties.core import PassiveParty
from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport

_METADATA = "party.metadata"
_LOCAL_LOGITS = "party.local_logits"
_LOCAL_GRADIENT = "party.local_gradient"


@dataclass
class RemotePartyService:
    """Expose party-local computations without moving the raw feature matrix.

    The service owns a normal :class:`PassiveParty`; callers receive only metadata
    and requested derived results. The raw ``_x`` array never appears in a response.
    This reference service intentionally starts with the operations required by the
    logistic VFL protocol. Histogram-tree RPC can be added behind the same boundary.
    """

    party: PassiveParty
    allowed_senders: set[str] | None = None

    def handle(self, message_type: str, sender_role: str, payload: Any) -> Any:
        if self.allowed_senders is not None and sender_role not in self.allowed_senders:
            raise ValueError(f"sender role {sender_role!r} is not authorized for this party")
        if message_type == _METADATA:
            return {
                "name": self.party.name,
                "n_rows": self.party.n_rows,
                "n_features": self.party.n_features,
            }
        if message_type == _LOCAL_LOGITS:
            weights, indices = self._pair(payload, _LOCAL_LOGITS)
            return self.party.local_logits(
                np.asarray(weights, dtype=float),
                self._indices(indices),
            )
        if message_type == _LOCAL_GRADIENT:
            residual, indices = self._pair(payload, _LOCAL_GRADIENT)
            return self.party.local_gradient(
                np.asarray(residual, dtype=float),
                self._indices(indices),
            )
        raise ValueError(f"unsupported remote party operation: {message_type}")

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
    """Client-side proxy for a passive party whose feature matrix is remote."""

    name: str
    n_rows: int
    n_features: int
    transport: RemoteHTTPTransport
    coordinator_role: str = "bank"

    @classmethod
    def discover(
        cls,
        name: str,
        transport: RemoteHTTPTransport,
        *,
        coordinator_role: str = "bank",
    ) -> RemotePassiveParty:
        metadata = transport.send(
            None,
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
        return cls(
            name=name,
            n_rows=int(metadata["n_rows"]),
            n_features=int(metadata["n_features"]),
            transport=transport,
            coordinator_role=coordinator_role,
        )

    def local_logits(
        self,
        weights: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        delivered = self.transport.send(
            (np.asarray(weights, dtype=float), indices),
            message_type=_LOCAL_LOGITS,
            sender_role=self.coordinator_role,
            receiver_role=self.name,
            direction="rpc",
            stage="party_compute",
        )
        return np.asarray(delivered, dtype=float)

    def local_gradient(
        self,
        residual: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        delivered = self.transport.send(
            (np.asarray(residual, dtype=float), indices),
            message_type=_LOCAL_GRADIENT,
            sender_role=self.coordinator_role,
            receiver_role=self.name,
            direction="rpc",
            stage="party_compute",
        )
        return np.asarray(delivered, dtype=float)
