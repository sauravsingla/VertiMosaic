from __future__ import annotations

import numpy as np

from vertimosaic.alignment import EntityAligner


def select_sanitized_case_study(
    entity_indices: np.ndarray,
    bank_only_probability: np.ndarray,
    full_vfl_probability: np.ndarray,
    party_permuted_probability: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    minimum_probability_shift: float = 0.05,
    salt: str = "vertimosaic-research-pseudonym",
) -> dict[str, object]:
    """Select a genuinely qualifying measured test example without raw features.

    A qualifying example must change the validation-selected binary decision between
    the Bank-only model and the four-party VFL model and must change predicted risk by
    at least ``minimum_probability_shift``. The function deliberately fails closed
    rather than substituting a weaker example when no such measured test entity exists.
    """
    entity_indices = np.asarray(entity_indices).reshape(-1)
    bank = np.asarray(bank_only_probability, dtype=float).reshape(-1)
    full = np.asarray(full_vfl_probability, dtype=float).reshape(-1)
    if not (len(entity_indices) == len(bank) == len(full)):
        raise ValueError("case-study arrays must have equal length")
    if not len(full):
        raise ValueError("case-study selection requires at least one test entity")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("case-study threshold must be in [0, 1]")
    if not 0.0 <= minimum_probability_shift <= 1.0:
        raise ValueError("minimum_probability_shift must be in [0, 1]")
    for party, probabilities in party_permuted_probability.items():
        if len(np.asarray(probabilities).reshape(-1)) != len(full):
            raise ValueError(f"permuted probabilities for {party} have the wrong length")

    difference = np.abs(full - bank)
    decision_changed = (bank >= threshold) != (full >= threshold)
    materially_changed = difference >= minimum_probability_shift
    candidates = np.flatnonzero(decision_changed & materially_changed)
    if not len(candidates):
        raise ValueError(
            "no measured test entity satisfies the strict case-study criterion: "
            "Bank-only and four-party VFL decisions must differ and the absolute "
            f"probability shift must be at least {minimum_probability_shift:.3f}"
        )

    local = int(np.argmax(difference[candidates]))
    selected = int(candidates[local])
    aligner = EntityAligner(salt)
    contributions = {
        party: float(full[selected] - np.asarray(probabilities, dtype=float)[selected])
        for party, probabilities in party_permuted_probability.items()
    }
    return {
        "entity_id": aligner.pseudonymize(str(entity_indices[selected])),
        "bank_only_risk": float(bank[selected]),
        "four_party_vfl_risk": float(full[selected]),
        "absolute_probability_shift": float(difference[selected]),
        "threshold": float(threshold),
        "minimum_probability_shift": float(minimum_probability_shift),
        "decision_changed": True,
        "qualifying_case": True,
        "selection_rule": (
            "largest_bank_to_full_shift_among_material_threshold_crossings"
        ),
        "contribution_summary": contributions,
        "contribution_method": "party_representation_permutation_probability_delta",
        "causal_importance": False,
    }
