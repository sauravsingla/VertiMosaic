# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hashlib import sha256
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class _PartyLike(Protocol):
    name: str

    @property
    def n_rows(self) -> int: ...


def _normalize_entity_ids(entity_ids: Sequence[object] | np.ndarray) -> np.ndarray:
    values = np.asarray(entity_ids, dtype=str).reshape(-1).copy()
    if values.size == 0:
        raise ValueError("entity_ids must not be empty")
    if np.any(np.char.str_len(values) == 0):
        raise ValueError("entity_ids must not contain empty values")
    if len(set(values.tolist())) != len(values):
        raise ValueError("entity_ids must be unique within a party partition")
    values.setflags(write=False)
    return values


def ordered_entity_digest(entity_ids: Sequence[object] | np.ndarray) -> str:
    """Return an order-sensitive digest for a sequence of entity identifiers."""
    values = _normalize_entity_ids(entity_ids)
    digest = sha256()
    digest.update(b"vertimosaic-entity-order-v1\0")
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def bind_entity_ids(party: _PartyLike, entity_ids: Sequence[object] | np.ndarray) -> None:
    """Attach immutable ordered entity identifiers and their digest to a party object.

    ``PassiveParty`` deliberately keeps raw features private.  Entity identifiers are
    therefore attached as protocol metadata rather than mixed into the feature matrix.
    The IDs may already be pseudonymous.  Only the order-sensitive digest is required by
    the model boundary; callers can drop the raw identifier array after binding if their
    deployment policy requires it.
    """
    values = _normalize_entity_ids(entity_ids)
    if len(values) != party.n_rows:
        raise ValueError(
            f"entity_ids length for party {party.name!r} ({len(values)}) does not match "
            f"its row count ({party.n_rows})"
        )
    setattr(party, "_entity_ids", values)
    setattr(party, "_entity_digest", ordered_entity_digest(values))


def entity_ids_for(party: _PartyLike) -> np.ndarray | None:
    values = getattr(party, "_entity_ids", None)
    if values is None:
        return None
    return np.asarray(values, dtype=str).reshape(-1)


def entity_digest_for(party: _PartyLike) -> str | None:
    digest = getattr(party, "_entity_digest", None)
    return None if digest is None else str(digest)


def validate_exact_entity_alignment(
    parties: Sequence[_PartyLike],
    *,
    context: str,
    require_bound_ids: bool = False,
) -> str | None:
    """Validate that parties refer to exactly the same entities in exactly the same order.

    When ``require_bound_ids`` is false, a collection in which *no* party has identifiers
    remains accepted for backwards compatibility.  A mixed collection (some bound, some
    unbound) is always rejected.  Production/research entry points should bind identifiers
    and use ``require_bound_ids=True``.
    """
    if not parties:
        raise ValueError(f"{context}: at least one party is required")
    digests = [entity_digest_for(party) for party in parties]
    present = [digest is not None for digest in digests]
    if not any(present):
        if require_bound_ids:
            names = ", ".join(party.name for party in parties)
            raise ValueError(
                f"{context}: explicit entity identifiers are required for parties: {names}; "
                "call bind_entity_ids(...) before training or inference"
            )
        return None
    if not all(present):
        missing = ", ".join(
            party.name for party, digest in zip(parties, digests, strict=True) if digest is None
        )
        raise ValueError(f"{context}: missing entity identifiers for parties: {missing}")
    first = digests[0]
    assert first is not None
    for party, digest in zip(parties[1:], digests[1:], strict=True):
        if digest != first:
            raise ValueError(
                f"{context}: entity order mismatch for party {party.name!r}; "
                "all parties must contain the same ordered entity identifiers"
            )
    return first
