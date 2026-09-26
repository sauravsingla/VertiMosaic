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
    salt: str = "vertimosaic-research-pseudonym",
) -> dict[str, object]:
    """Select one measured test example without exposing row-level feature values."""
    entity_indices = np.asarray(entity_indices).reshape(-1)
    bank = np.asarray(bank_only_probability, dtype=float).reshape(-1)
    full = np.asarray(full_vfl_probability, dtype=float).reshape(-1)
    if not (len(entity_indices) == len(bank) == len(full)):
        raise ValueError("case-study arrays must have equal length")
    if not len(full):
        raise ValueError("case-study selection requires at least one test entity")
    for party, probabilities in party_permuted_probability.items():
        if len(np.asarray(probabilities).reshape(-1)) != len(full):
            raise ValueError(f"permuted probabilities for {party} have the wrong length")
    difference = np.abs(full - bank)
    threshold_crossing = (bank < threshold) != (full < threshold)
    candidates = np.flatnonzero(threshold_crossing)
    if len(candidates):
        local = int(np.argmax(difference[candidates]))
        selected = int(candidates[local])
        selection_rule = "largest_bank_to_full_shift_among_threshold_crossings"
    else:
        selected = int(np.argmax(difference))
        selection_rule = "largest_bank_to_full_probability_shift"
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
        "selection_rule": selection_rule,
        "contribution_summary": contributions,
        "contribution_method": "party_representation_permutation_probability_delta",
        "causal_importance": False,
    }
