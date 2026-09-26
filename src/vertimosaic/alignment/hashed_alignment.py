"""Research pseudonymous alignment utilities."""

from __future__ import annotations

import hashlib


def pseudonymize(value: str, salt: str) -> str:
    """Return a deterministic research pseudonym; this is not PSI."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def intersect_ids(id_sets: dict[str, set[str]]) -> set[str]:
    if not id_sets:
        return set()
    return set.intersection(*id_sets.values())
