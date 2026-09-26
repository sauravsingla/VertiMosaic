from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import factorial

PartyScore = Callable[[tuple[str, ...]], float]


def enumerate_party_subsets(parties: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [subset for r in range(len(parties) + 1) for subset in combinations(parties, r)]


def exact_shapley_party_utility(
    parties: tuple[str, ...], score_subset: PartyScore
) -> dict[str, float]:
    """Exact predictive party utility for a small party set; not causal importance."""
    n = len(parties)
    if n == 0:
        return {}
    utilities = {party: 0.0 for party in parties}
    for party in parties:
        others = tuple(item for item in parties if item != party)
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for subset in combinations(others, r):
                with_party = tuple(sorted((*subset, party)))
                base = tuple(sorted(subset))
                utilities[party] += weight * (score_subset(with_party) - score_subset(base))
    return utilities
