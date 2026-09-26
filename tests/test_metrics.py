import numpy as np

from vertimosaic.evaluation import bootstrap_metric_ci, evaluate_binary


def test_metrics_and_ci() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6])
    m = evaluate_binary(y, p)
    assert m.roc_auc == 1.0
    low, high = bootstrap_metric_ci(y, p, replicates=50, seed=1)
    assert 0 <= low <= high <= 1
