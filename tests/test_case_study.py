import numpy as np
import pytest

from vertimosaic.reporting import select_sanitized_case_study


def _permuted() -> dict[str, np.ndarray]:
    return {
        "bank": np.array([0.25, 0.55, 0.60, 0.58]),
        "telecom": np.array([0.28, 0.60, 0.64, 0.59]),
        "insurance": np.array([0.29, 0.68, 0.62, 0.57]),
        "retail": np.array([0.31, 0.70, 0.66, 0.61]),
    }


def test_case_study_selects_strict_measured_example_without_raw_features() -> None:
    entity_indices = np.array([10, 11, 12, 13])
    bank = np.array([0.2, 0.4, 0.7, 0.8])
    full = np.array([0.3, 0.75, 0.65, 0.6])
    result = select_sanitized_case_study(
        entity_indices,
        bank,
        full,
        _permuted(),
        threshold=0.5,
        minimum_probability_shift=0.05,
    )
    assert len(str(result["entity_id"])) == 64
    assert result["selection_rule"] == (
        "largest_bank_to_full_shift_among_material_threshold_crossings"
    )
    assert result["qualifying_case"] is True
    assert result["decision_changed"] is True
    assert float(result["absolute_probability_shift"]) >= 0.05
    assert set(result["contribution_summary"]) == {"bank", "telecom", "insurance", "retail"}
    assert "raw_features" not in result
    assert result["causal_importance"] is False


def test_case_study_fails_closed_without_threshold_crossing() -> None:
    entity_indices = np.array([10, 11, 12, 13])
    bank = np.array([0.2, 0.3, 0.7, 0.8])
    full = np.array([0.25, 0.35, 0.65, 0.75])
    with pytest.raises(ValueError, match="no measured test entity satisfies"):
        select_sanitized_case_study(
            entity_indices,
            bank,
            full,
            _permuted(),
            threshold=0.5,
        )


def test_case_study_fails_closed_when_crossing_is_not_material() -> None:
    entity_indices = np.array([10, 11, 12, 13])
    bank = np.array([0.2, 0.49, 0.7, 0.8])
    full = np.array([0.3, 0.51, 0.65, 0.6])
    with pytest.raises(ValueError, match="probability shift must be at least"):
        select_sanitized_case_study(
            entity_indices,
            bank,
            full,
            _permuted(),
            threshold=0.5,
            minimum_probability_shift=0.05,
        )


def test_case_study_validates_materiality_parameter() -> None:
    entity_indices = np.array([10])
    bank = np.array([0.4])
    full = np.array([0.6])
    permuted = {name: np.array([0.5]) for name in ("bank", "telecom", "insurance", "retail")}
    with pytest.raises(ValueError, match="minimum_probability_shift"):
        select_sanitized_case_study(
            entity_indices,
            bank,
            full,
            permuted,
            minimum_probability_shift=1.1,
        )
