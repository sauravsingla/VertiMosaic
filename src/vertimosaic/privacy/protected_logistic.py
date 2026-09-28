# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vertimosaic.alignment import bind_entity_ids
from vertimosaic.models.vfl_logistic import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.privacy.backends import ClippedGaussianDPBackend


class PSIBackend(Protocol):
    def intersection(self, client_ids: list[str], server_ids: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class ProtectedLogisticRun:
    """Artifacts returned by the explicitly protected logistic research path."""

    model: VFLLogisticRegression
    active: ActiveParty
    passive: tuple[PassiveParty, ...]
    entity_ids: tuple[str, ...]

    def privacy_report(self, *, delta: float) -> dict[str, object]:
        residual_report = self.model.privacy_report(delta=delta)
        return {
            "entity_alignment": "PSI intersection supplied by the configured PSI backend",
            "residual_release": residual_report,
            "guarantee_boundary": (
                "PSI protects set intersection according to its backend threat model; "
                "clipped Gaussian accounting covers active-to-passive residual releases only. "
                "Logits, model parameters, timing, other-model routing, and transport metadata "
                "remain outside this composite guarantee."
            ),
        }


def _unique_ids(values: Sequence[object], *, party_name: str) -> list[str]:
    normalized = [str(value) for value in values]
    if not normalized:
        raise ValueError(f"{party_name}: entity IDs must not be empty")
    if any(not value for value in normalized):
        raise ValueError(f"{party_name}: entity IDs must not contain empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{party_name}: entity IDs must be unique")
    return normalized


def psi_align_parties(
    active: ActiveParty,
    active_entity_ids: Sequence[object],
    passive: Sequence[PassiveParty],
    passive_entity_ids: Sequence[Sequence[object]],
    *,
    psi_backend: PSIBackend,
) -> tuple[ActiveParty, list[PassiveParty], tuple[str, ...]]:
    """Intersect entity sets with PSI and return identically ordered party partitions."""
    if len(passive) != len(passive_entity_ids):
        raise ValueError("passive parties and passive_entity_ids must have equal length")
    active_ids = _unique_ids(active_entity_ids, party_name=active.name)
    if len(active_ids) != active.n_rows:
        raise ValueError("active entity ID count must match active rows")

    passive_ids: list[list[str]] = []
    for party, identifiers in zip(passive, passive_entity_ids, strict=True):
        normalized = _unique_ids(identifiers, party_name=party.name)
        if len(normalized) != party.n_rows:
            raise ValueError(f"{party.name}: entity ID count must match party rows")
        passive_ids.append(normalized)

    common = active_ids.copy()
    for identifiers in passive_ids:
        common = psi_backend.intersection(common, identifiers)
    if not common:
        raise ValueError("PSI intersection is empty")

    common_set = set(common)
    ordered_common = tuple(value for value in active_ids if value in common_set)
    if not ordered_common:
        raise ValueError("PSI intersection is empty after canonical ordering")

    active_lookup = {value: index for index, value in enumerate(active_ids)}
    active_index = np.asarray([active_lookup[value] for value in ordered_common], dtype=int)
    aligned_active = ActiveParty(
        active.name,
        active._x[active_index],
        active.labels[active_index],
    )
    bind_entity_ids(aligned_active, ordered_common)

    aligned_passive: list[PassiveParty] = []
    for party, identifiers in zip(passive, passive_ids, strict=True):
        lookup = {value: index for index, value in enumerate(identifiers)}
        missing = [value for value in ordered_common if value not in lookup]
        if missing:
            raise ValueError(
                f"PSI backend returned entities not present in party {party.name!r}: {missing[:3]}"
            )
        index = np.asarray([lookup[value] for value in ordered_common], dtype=int)
        aligned = PassiveParty(party.name, party._x[index])
        bind_entity_ids(aligned, ordered_common)
        aligned_passive.append(aligned)
    return aligned_active, aligned_passive, ordered_common


def fit_protected_logistic(
    active: ActiveParty,
    active_entity_ids: Sequence[object],
    passive: Sequence[PassiveParty],
    passive_entity_ids: Sequence[Sequence[object]],
    *,
    psi_backend: PSIBackend,
    clip_l2_norm: float = 1.0,
    noise_multiplier: float = 2.0,
    adjacency: str = "replace_one",
    seed: int = 42,
    model_kwargs: dict[str, object] | None = None,
) -> ProtectedLogisticRun:
    """Run PSI alignment plus clipped-Gaussian residual-release VFL logistic training.

    This is an explicitly scoped protected research configuration, not a generic
    "secure VFL" mode. The PSI backend protects entity-set intersection according to
    its own threat model, while the DP backend covers only residual messages sent from
    the active party to passive parties.
    """
    aligned_active, aligned_passive, entity_ids = psi_align_parties(
        active,
        active_entity_ids,
        passive,
        passive_entity_ids,
        psi_backend=psi_backend,
    )
    backend = ClippedGaussianDPBackend(
        clip_l2_norm=clip_l2_norm,
        noise_multiplier=noise_multiplier,
        adjacency=adjacency,
        seed=seed,
    )
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("seed", seed)
    kwargs.setdefault("require_entity_ids", True)
    kwargs.setdefault("missing_party_policy", "error")
    kwargs["residual_dp_backend"] = backend
    model = VFLLogisticRegression(**kwargs)
    model.fit(aligned_active, aligned_passive)
    return ProtectedLogisticRun(
        model=model,
        active=aligned_active,
        passive=tuple(aligned_passive),
        entity_ids=entity_ids,
    )
