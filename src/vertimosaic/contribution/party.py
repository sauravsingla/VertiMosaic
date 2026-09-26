"""Party-level predictive utility helpers."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations


def powerset_parties(parties: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [subset for r in range(len(parties) + 1) for subset in combinations(parties, r)]


def exact_shapley_utility(
    parties: tuple[str, ...],
    utility: Callable[[tuple[str, ...]], float],
) -> dict[str, float]:
    """Exact Shapley utility over a small party set.

    This is predictive utility, not causal importance.
    """
    import math

    n = len(parties)
    values: dict[str, float] = {}
    for party in parties:
        others = tuple(p for p in parties if p != party)
        total = 0.0
        for subset in powerset_parties(others):
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n)
            with_party = tuple(sorted((*subset, party)))
            total += weight * (utility(with_party) - utility(tuple(sorted(subset))))
        values[party] = total
    return values
