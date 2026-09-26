from vertimosaic.experiments import run_demo


def test_demo_logistic_smoke() -> None:
    metrics = run_demo(rows=500, seed=42, model_name="logistic")
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["estimated_communication_bytes"] > 0
