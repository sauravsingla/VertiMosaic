import numpy as np

from vertimosaic.reporting import select_sanitized_case_study


def test_case_study_selects_measured_example_without_raw_features() -> None:
    entity_indices = np.array([10, 11, 12, 13])
    bank = np.array([0.2, 0.4, 0.7, 0.8])
    full = np.array([0.3, 0.75, 0.65, 0.6])
    permuted = {
        "bank": np.array([0.25, 0.55, 0.60, 0.58]),
        "telecom": np.array([0.28, 0.60, 0.64, 0.59]),
        "insurance": np.array([0.29, 0.68, 0.62, 0.57]),
        "retail": np.array([0.31, 0.70, 0.66, 0.61]),
    }
    result = select_sanitized_case_study(entity_indices, bank, full, permuted, threshold=0.5)
    assert len(str(result["entity_id"])) == 64
    assert result["selection_rule"] == "largest_bank_to_full_shift_among_threshold_crossings"
    assert set(result["contribution_summary"]) == {"bank", "telecom", "insurance", "retail"}
    assert "raw_features" not in result
    assert result["causal_importance"] is False
