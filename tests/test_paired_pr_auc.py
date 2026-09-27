import numpy as np

from vertimosaic.evaluation import paired_bootstrap_difference
from vertimosaic.reporting.interpreter import comparison_observation


def test_roc_comparison_also_reports_paired_pr_auc_on_same_resamples() -> None:
    y = np.array([0, 1] * 50)
    probabilities = np.linspace(0.05, 0.95, 100)
    comparison = paired_bootstrap_difference(
        y,
        probabilities,
        probabilities,
        metric="roc_auc",
        replicates=50,
        seed=31,
    )
    assert comparison["delta"] == 0.0
    assert comparison["paired_pr_auc_delta"] == 0.0
    assert comparison["paired_pr_auc_lower"] == 0.0
    assert comparison["paired_pr_auc_upper"] == 0.0
    text = comparison_observation(comparison)
    assert "roc_auc" in text
    assert "pr_auc" in text
